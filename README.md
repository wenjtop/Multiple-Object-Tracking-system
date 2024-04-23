# Multiple-Object-Tracking-system
系统支持：多目标实时跟踪，上传视频跟踪、视频管理、目标计数、跟踪截图报告、跟踪结果下载、远程域名访问。 多目标跟踪算法检测部分是：YOLOv8。数据关联部分：重新设计的一个算法。系统框架：Django。系统做了内网穿透支持远程域名访问。视频实时推流：重新写了一个视频推流方法，延迟能做到300ms左右，基本能满足实时跟踪，偶尔会出现掉帧现象。