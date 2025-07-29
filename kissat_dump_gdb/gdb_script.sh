# gdb_commands.txt
# 这个脚本由 GDB 自动执行

# 1. 加载我们的 Python 脚本，以定义自定义命令 'dump_solver_attrs'
source ./dump_state.py

# 2. 在目标位置设置断点
# 请确保这里的路径和行号是正确的
break src/decide.c:135

# 3. 为刚刚创建的断点 1 定义自动执行的命令
commands 1
  # silent: 保持输出干净，不显示命令本身
  silent
  # 调用 Python 脚本中定义的命令来 dump 属性
  dump_solver_attrs
  # 继续程序运行
  continue
end

# 4. 运行程序
# 'run' 命令会自动使用通过 '--args' 传入的程序参数
run

# 5. 程序执行结束后，自动退出 GDB
# 如果你想在程序结束后停留在 GDB 中进行检查，可以注释掉下面这行
quit