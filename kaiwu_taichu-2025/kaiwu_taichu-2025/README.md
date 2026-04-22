# 开悟2025比赛

> 本仓库同时包含王者和具身的比赛代码
>
> 2024年王者初赛代码[跳转分支](https://github.com/wty-yy/kaiwu_taichu/tree/2024)
>
> 开悟训练框架说明可参考[Blog - 2024开悟智能体比赛（海选赛）](https://wty-yy.github.io/posts/33143/#localremote%E8%AE%AD%E7%BB%83%E9%80%BB%E8%BE%91)

| 仓库名 | 介绍 | 名次 |
| - | - | - |
| [`hok_prelim`](./hok_prelim/) | 王者(高级)初赛 [笔记](https://wty-yy.github.io/posts/19715/) | 41 |
| [`rob_prelim`](./rob_prelim/) | 具身初赛 | 11 |
| [`hok_semi`](./hok_semi/) | 王者(高级)复赛 | 1 |
| `hok_final` (未开源) | 王者(高级)决赛| 4 |
| [`rob_final`](https://github.com/wty-yy/go2_rl_gym) | 具身决赛 | 1 |

> hok_final成绩较差, 代码实现应该存在问题, 仍需改进, 先不开源了
>
> rob_final代码全部在isaacgym中实现，开源在[go2_rl_gym](https://github.com/wty-yy/go2_rl_gym)，包含训练、验证、部署代码


## 王者/具身初赛
两个比赛题目基本一致，但是版本名和版本上存在小差别
- `hok_prelim`: 智能体赛道代码
- `rob_prelim`: 机器人赛道代码
> 在腾讯开悟客户端中选择工作路径分别为`/path/to/your/hok_prelim`和`/path/to/your/rob_prelim`即可

两个赛道初赛的区别：
- 具身视野范围51x51, 王者视野范围11x11
- 具身完全看不到视野外的宝箱、终点、buff位置，王者可以看到视野外宝箱、终点、buff的大致位置

最后比赛的配置均为: 随机8个宝箱、buff、起点、终点位置，随机障碍物，最大步数2000

更新日志见[CHANGELOG.md](./CHANGELOG.md)

## 王者复赛
日志请见[hok_semi/README.md](./hok_semi/README.md)

# 加载已有模型方法
以具身赛道为例
1. 将训练保存的`backup_model/`文件夹下的`*.zip`压缩包解压出来, 找到`ckpt/model.ckpt-*.pkl`文件, 复制到[`code/ckpt`](./rob_prelim/code/ckpt)下
2. 修改[`conf/configure_app.toml`](./rob_prelim/code/conf/configure_app.toml)文件中的`preload_model = true`, `preload_model_id = *`（这个`*`就是上面`model.ckpt-*.pkl`中的数字）
3. 正常启动训练即可, 在训练日志的`aisrv.log`中可以看到`First load model <model_file_path> and update target q successfully`就说明加载成功了
