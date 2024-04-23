系统支持：多目标实时跟踪，上传视频跟踪、视频管理、目标计数、跟踪截图报告、跟踪结果下载、远程域名访问。 多目标跟踪算法检测部分是：YOLOv8。数据关联部分：重新设计的一个算法。系统框架：Django。系统做了内网穿透支持远程域名访问。视频实时推流：重新写了一个视频推流方法，延迟能做到300ms左右，基本能满足实时跟踪，偶尔会出现掉帧现象。
![Uploading image.png…]()

(1) 采用最新的YOLOv8作为多目标跟踪的检测器，并重新设计了一种多目标跟踪的数据关联算法，命名为EnhanceSort。该算法主要思想是保留中高分检测框，并将它们与正在跟踪的轨迹、新建立的轨迹和丢失的轨迹依次进行匹配，形成更有效的匹配策略。其中，引入中分检测框与轨迹相关联，可以恢复正确检测的中分检测框并过滤背景。在轨迹预测中，将卡尔曼滤波的状态表示从XYAH（中心坐标X、中心坐标Y、高宽比A、高度H）改为XYWH（中心坐标X、中心坐标Y、宽度W、高度H），有助于简化跟踪过程并使目标的空间尺寸更直观易懂。这可以减少计算过程中产生的误差和冗余，同时提高算法的计算效率和跟踪准确性，为多目标跟踪算法的优化提供帮助。在衡量轨迹预测框与检测框距离时，使用GIOU替代IOU作为相似度度量。GIOU考虑了目标边界框之间的几何关系，包括宽高比和相对位置，从而在目标跟踪中提供更准确的匹配信息。这提高了跟踪算法在处理遮挡、目标间交互等复杂场景时的准确性和鲁棒性，特别是在处理复杂场景和遮挡问题时具有显著优势。
(2) 本文重新设计了一种多目标检测跟踪的特征重识别算法，命名为LNG-Transformer。该算法的每个网络模块都融合了局部信息、邻域信息和全局信息的交互。在网络的任何时期，该算法都能够学习到局部信息和全局信息。从而能够有效地提取目标的深层特征，实现更为精确的目标关联。
(3) 设计了多目标跟踪系统，该系统包括用户注册、登录及用户信息管理模块，摄像头实时多目标跟踪模块和数据管理模块。为计算机视觉、智能安全监控和商业客流分析等领域提供了有效的解决方案。


演示视频：https://www.bilibili.com/video/BV1vP411B7yU/?spm_id_from=333.999.0.0&vd_source=988e8e87fabf17ffc54af71ef26fa298
