import argparse
from rich import print,console
from pathlib import Path
import os
import hydra
import ray

from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import (
    SpeedTestCallback, 
    RecordCallback, 
    RewardsCallback, 
    TaskCallback, 
    FastResetCallback, 
    InitInventoryCallback,
    SummonMobsCallback,
    CommandsCallback,
    # TeleportCallback,
)
from minestudio.models import CraftWorker,SmeltWorker

from jarvisvla.evaluate import draw_utils
from jarvisvla.utils import file_utils
from jarvisvla.evaluate import agent_wrapper


def evaluate(video_path,checkpoints,environment_config:dict,model_config:dict,device="cuda:0",base_url=None):
    """
    执行单个评估任务。

    Args:
        video_path (str): 视频输出路径。
        checkpoints (str): 模型检查点路径。
        environment_config (dict): 环境配置，包含env_config, max_frames, verbos等。
        model_config (dict): 模型配置，包含temperature, history_num, instruction_type, action_chunk_len等。
        device (str, optional): 运行设备的名称，例如 "cuda:0"。默认为 "cuda:0"。
        base_url (str, optional): VLLM服务的API基础URL。如果为None，则表示不使用VLLM服务。

    Returns:
        tuple: (success, frames_taken) 如果任务成功，返回True和完成任务的帧数；否则返回False和-1。
    """
    # 清理 Hydra 的全局实例，确保配置加载的独立性
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # 根据环境配置获取配置文件路径和名称
    config_path = Path(f"{environment_config['env_config']}.yaml")
    config_name = config_path.stem
    config_path = os.path.join("./config",config_path.parent)
    # 初始化Hydra配置
    hydra.initialize(config_path=config_path, version_base='1.3')
    cfg = hydra.compose(config_name=config_name)
    
    # 配置相机参数
    camera_cfg = CameraConfig(**cfg.camera_config)
    # 配置视频录制回调函数
    record_callback = RecordCallback(record_path=Path(video_path).parent, fps=30,show_actions=False)  
    # 定义一系列回调函数，用于控制环境行为和数据收集
    callbacks = [
        FastResetCallback(
            biomes=cfg.candidate_preferred_spawn_biome,
            random_tp_range=cfg.random_tp_range,
            start_time=cfg.start_time,
        ), 
        SpeedTestCallback(50), 
        TaskCallback(getattr(cfg,"task_conf",None)),
        RewardsCallback(getattr(cfg,"reward_conf",None)),
        InitInventoryCallback(cfg.init_inventory,
                                inventory_distraction_level=cfg.inventory_distraction_level,
                                equip_distraction_level="normal"
                                ),
        CommandsCallback(getattr(cfg,"command",[]),),
        record_callback,
    ]
    # 如果配置中包含mobs，则添加召唤怪物回调
    if cfg.mobs:
        callbacks.append(SummonMobsCallback(cfg.mobs))
    
    # 初始化Minecraft模拟器环境
    env =  MinecraftSim(
        action_type="env", # 初始动作为环境控制
        seed=cfg.seed,
        obs_size=cfg.origin_resolution,
        render_size=cfg.resize_resolution,
        camera_config=camera_cfg,
        preferred_spawn_biome=getattr(cfg,"preferred_spawn_biome",None),
        callbacks = callbacks
    )
    # 重置环境并获取初始观察和信息
    obs, info = env.reset()

    # 初始化代理
    agent = None
    pre_agent = None
    worker_type =  getattr(cfg,"worker", None)
    # 根据配置选择预处理代理（如合成或冶炼）
    if worker_type == "craft":
        pre_agent = CraftWorker(env,if_discrete=True)
    elif worker_type == "smelt":
        pre_agent = SmeltWorker(env,if_discrete=True)
    
    # 准备环境，例如打开工作台或熔炉
    need_crafting_table = False
    if getattr(cfg, "need_gui", False):
        need_crafting_table= getattr(cfg,"need_crafting_table", False)
        need_furnace = getattr(cfg,"need_furnace", False)
        if need_crafting_table:
            try:
                frames,_,_ = pre_agent.open_crating_table_wo_recipe()
            except AssertionError as e:
                env.close()
                console.Console().log(f"error: {e}")
                return False,-1
        elif need_furnace:
            try:
                frames,_,_ = pre_agent.open_furnace_wo_recipe()
            except AssertionError as e:
                env.close()
                console.Console().log(f"error: {e}")
                return False,-1
        else:
            pre_agent._null_action(1)
            if not pre_agent.info['isGuiOpen']:
                pre_agent._call_func('inventory')
        
    env.action_type = "agent"  # 将动作类型切换为代理控制

    # 根据base_url是否存在，初始化VLLM代理或抛出异常
    if type(base_url)!=type(None):
        agent = agent_wrapper.VLLM_AGENT(checkpoint_path=checkpoints,base_url=base_url,**model_config)
    else:
        raise Exception("base_url 不能为空，请提供 VLLM 服务的 URL。")
        
    # 获取任务指令
    instructions = [item["text"] for item in cfg.task_conf]

    success = (False,environment_config["max_frames"])
    # 在最大帧数内循环执行代理动作
    for i in range(environment_config["max_frames"]):
        # 代理根据当前观察和指令生成动作
        action = agent.forward([info["pov"]],instructions,verbos=environment_config["verbos"],need_crafting_table = need_crafting_table)
        if environment_config["verbos"]:
            console.Console().log(action)
        # 环境执行动作并返回新的观察、奖励、终止状态等
        obs, reward, terminated, truncated, info = env.step(action)

        # 如果获得奖励，表示任务成功，提前结束循环
        if reward>0:
            success = (True,i)
            break   
        
    # 如果任务成功，额外采样20步以完成视频录制或后续操作
    if success[0]:
        for i in range(20):
            action = agent.forward([info["pov"]],instructions,verbos=environment_config["verbos"],need_crafting_table = need_crafting_table)
            obs, reward, terminated, truncated, info = env.step(action)
         
    # 关闭环境
    env.close()
    return success

@ray.remote
def evaluate_wrapper(video_path,checkpoints,environment_config,base_url,model_config):
    """
    Ray远程调用的评估包装函数。
    Args:
        video_path (str): 视频输出路径。
        checkpoints (str): 模型检查点路径。
        environment_config (dict): 环境配置。
        base_url (str): VLLM服务的API基础URL。
        model_config (dict): 模型配置。
    Returns:
        tuple: (success, frames_taken, member_id) 任务成功状态、帧数和成员ID。
    """
    success = evaluate(video_path=video_path,checkpoints=checkpoints,environment_config=environment_config,base_url=base_url,model_config=model_config)
    member_id = video_path.split("/")[-1].split(".")[0]
    return success[0],success[1],member_id

def multi_evaluate(args):
    """
    执行多任务并行评估。

    Args:
        args (argparse.Namespace): 命令行参数，包含workers, split_number, env_config等。
    """
    ray.init() # 初始化Ray
    import os
    from pathlib import Path
    
    # 根据检查点路径生成模型引用名称，用于日志和视频文件夹命名
    model_ref_name = args.checkpoints.split('/')[-1]
    if "checkpoint" in model_ref_name:
        checkpoint_num = model_ref_name.split("-")[-1]
        model_base_name = args.checkpoints.split('/')[-2]
        model_ref_name = f"{model_base_name}-{checkpoint_num}"
    
    # 构建视频输出主文件夹路径
    video_fold  = os.path.join(args.video_main_fold, f"{model_ref_name}-{args.env_config.split('/')[-1]}") 
    # 如果文件夹不存在则创建
    if not os.path.exists(video_fold):
        Path(video_fold).mkdir(parents=True,exist_ok=True)
    
    # 准备模型和环境配置字典
    model_config = dict(
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    
    # 加载已有的评估结果日志
    video_log_path = os.path.join(video_fold,"end.json") 
    resultss = file_utils.load_json_file(video_log_path,data_type="list")

    # 确定需要执行的任务ID
    total_ids = [i for i in range(args.workers)]
    done_ids = [results[2] for results in resultss]
    undone_ids = [id for id in total_ids if str(id) not in done_ids]

    if not undone_ids:
        return # 如果所有任务都已完成，则直接返回
    
    # 分批次执行未完成的任务
    roll = len(undone_ids) // args.split_number + (1 if len(undone_ids) % args.split_number != 0 else 0)
    for i in range(roll):
        part_undone_ids = undone_ids[i*args.split_number:min((i+1)*args.split_number, len(undone_ids))]
        # 提交Ray远程任务
        result_ids = [evaluate_wrapper.remote(video_path=os.path.join(video_fold,str(i),f"{i}.mp4"),checkpoints=args.checkpoints,environment_config=environment_config,base_url=args.base_url,model_config=model_config) for i in part_undone_ids]
        futures = result_ids
        
        # 等待所有任务完成
        while len(futures) > 0:
            ready_futures, rest_futures = ray.wait(futures,timeout=24*60*60)
            results = ray.get(ready_futures,timeout=60*60)  # 获取结果
            resultss.extend(results)
            print(f"part frames IDs: {results} done!")
            futures = rest_futures
        
        ray.shutdown() # 关闭Ray
        
        # 写入日志文件
        file_utils.dump_json_file(resultss,video_log_path)
    # 显示成功率并保存图像
    draw_utils.show_success_rate(resultss,os.path.join(video_fold,"image.png") )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JarvisVLA评估脚本")
    # 定义命令行参数
    parser.add_argument('--workers', type=int, default=1, help='并行评估的worker数量。0表示不使用Ray并行，直接运行evaluate函数；1表示单worker运行evaluate函数；大于1表示使用Ray并行运行multi_evaluate函数。') 
    parser.add_argument('--split-number', type=int, default=6, help='在多worker模式下，每次Ray提交的任务批次大小。') 
    parser.add_argument('--env-config',"-e", type=str, default='craft/craft_bread', help='环境配置文件的路径，例如 craft/craft_bread。') 
    parser.add_argument('--max-frames', type=int, default=200, help='每个评估任务的最大帧数。') 
    parser.add_argument('--verbos', type=bool, default=False, help='是否启用详细输出。')
    parser.add_argument('--checkpoints', type=str, default="/public/models/qwen2-vl-7b-instruct/", help='模型检查点或模型文件的本地路径。')
    parser.add_argument('--device',type=str,default="cuda:0", help='模型运行的设备，例如 cuda:0。')
    
    parser.add_argument('--base-url',type=str, help='VLLM服务的API基础URL，例如 http://localhost:8000。')
    parser.add_argument('--video-main-fold',type=str, help='视频和日志文件的主输出文件夹。')
    
    parser.add_argument('--instruction-type',type=str,default='normal', help='指令类型，例如 normal。')
    parser.add_argument('--temperature','-t',type=float,default=0.7, help='模型推理的温度参数。')
    parser.add_argument('--history-num',type=int,default=0, help='用于推理的历史记录数量。')
    parser.add_argument('--action-chunk-len',type=int,default=1, help='动作块的长度。')

    args = parser.parse_args()

    # 再次构建模型和环境配置字典，确保所有参数都被正确传递
    model_config = dict(
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    # 如果base_url未提供，则设置为None
    if not args.base_url:
        args.base_url=None
    
    # 根据worker数量选择执行模式
    if args.workers==0:
        # 单独运行evaluate函数，不使用Ray并行，通常用于调试
        environment_config["verbos"] = True # 强制开启详细输出
        video_path = f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=video_path,checkpoints = args.checkpoints,environment_config = environment_config,device=args.device,model_config=model_config)
    elif args.workers==1:
        # 单worker运行evaluate函数，通常用于生产环境的单任务评估
        video_path = f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4",checkpoints = args.checkpoints,environment_config = environment_config,base_url=args.base_url,model_config=model_config)
    elif args.workers>1:
        # 使用Ray并行运行multi_evaluate函数，用于大规模评估
        multi_evaluate(args)
        
    
