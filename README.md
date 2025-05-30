# 小行星的形成

* 这一项目是2025年春季北京大学数学模型课程的第二次小组作业，作业内容即如标题所示，采用模拟等方法来定性/定量的讨论小行星的形成。

* 项目中实现了这个系统的数值模拟，使用LeapFrog积分器来模拟这个系统的动力学过程，同时提供了较为丰富的可视化和数据处理选项。

## 如何运行该项目

这一项目的代码中同时包括C++的部分和python的部分，我们在python中完成整体的控制逻辑，在C++中实现部分对性能要求较高的计算（如碰撞检测、加速度计算等），再使用pybind11将C++的代码编译为一个python包，从而实现两种语言之间的交互。因此，这一项目需要按如下的方式运行：

python环境可以借助于conda_requirements.txt配置，这一步中所安装的pybind11是下面C++的代码可以接入python中的前置条件，您应该确保这个包被安装，并且这一环境在终端中被激活。

* 如果不希望编译C++的代码：那么您可以在src/physics.py中注释掉对应于cpp_nbody_lib的import后直接进入运行的部分

* 如果您希望编译C++的代码，但是没有Cuda Toolkit或是Cuda Toolkit与您的编译器并未正确连接：

  那么您可以在src/computation/CMakeLists.txt中将第3行的CUDA删去，即修改为
  ```
  project(NBodyPythonExt LANGUAGES CXX)
  ```
  第60行将ON改成OFF，即修改为
  ```
  option(PROJECT_ENABLE_CUDA "Build with CUDA support if toolkit is found" OFF)
  ```

  此后正常进行cmake后编译，您会在src/computation/build文件夹下面看到对应的编译出的文件，之后您可以就运行python的部分了。

* 如果您的Cuda Toolkit存在并且与编译器正确连接：

  您可以直接使用src/computation/CMakeLists.txt，cmake后编译，您会在src/computation/build文件夹下面看到对应的编译出的文件，之后您可以就运行python的部分了。

* 其余问题：

  我们在MacOS下观察到OpenMP可能无法使用，在这种情况下您可以将对应C++代码中关于omp的行注释掉，此后可以同上进行。

在正确的将C++的代码编译成可以import的包之后，python代码可以以如下方式运行：

每组运行参数应该被放置于initial_conditions/1.json文件中，在列表后添加即可，这里建议让其id项与其余运行参数不同，否则默认的输出路径会覆盖掉先前的数据。运行参数的可以参考其余运行参数，其中涉及到的质量以太阳（恒星）质量为单位，长度以天文单位作为单位，从而时间单位是`1/(2*pi)`年，参数中的'eta'用于确定动态步长，通常应取在[0, 1)之间，但是如果希望使用默认步长的话，可以将其调成一个很大的数使动态步长无效化。

此后，在本项目的根目录下运行`python -m src.main`将src作为一个python包来运行。

此时您可以选择输入：

* `sim i j`其中`i`，`j`是两个整数，此时程序将检查initial_conditions/1.json中所有的设定中'id'在`i` `j`之间的运行参数，并且依次运行这一参数。运行结果按照默认的命名方式放置于visualization_data目录下。对于'id'为`setting_index`的运行参数而言，其所产生的文件分别为`./visualization_data/{run_index}-{setting_index}-data.dat`（记录了用于制作下面可视化界面的数据），`./visualization_data/{run_index}-{setting_index}-mass_histogram.png`（记录了随时间演化的质量直方图），`./visualization_data/{run_index}-{setting_index}-log_log.png`（记录了随时间演化的对数质量-对数数目分布）,`./visualization_data/{run_index}-{setting_index}-density_surface.png`（记录了粒子局部密度的分布）

* `show i`，假定`./visualization_data/1-{i}-data.dat`已经存在，那么可视化程序将会读取其中的数据并生成一个可以拖动的类似于视频的可视化窗口。

虽然上述描述中运行路径基本确定，但相关的修改如果有需要是并不困难的。



