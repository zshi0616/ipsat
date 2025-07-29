#!/bin/bash

# =================================================================
# GDB 自动化调试脚本
# =================================================================

# 检查所需文件是否存在
if [ ! -f "./build/kissat" ]; then
    echo "错误: 找不到可执行文件 ./build/kissat"
    exit 1
fi
if [ ! -f "./e10.cnf" ]; then
    echo "错误: 找不到输入文件 ./e10.cnf"
    exit 1
fi
if [ ! -f "./gdb_script.sh" ]; then
    echo "错误: 找不到 GDB 命令脚本 ./gdb_script.sh"
    exit 1
fi
if [ ! -f "./dump_state.py" ]; then
    echo "错误: 找不到 Python 脚本 ./dump_state.py"
    exit 1
fi


echo "启动 GDB 自动化调试..."
echo "程序: ./build/kissat"
echo "参数: ./e10.cnf"
echo "-------------------------------------"

# 使用 GDB 的 -x 选项来执行命令文件
# 使用 --args 来分割 GDB 的参数和要调试程序的参数
gdb -x ./gdb_script.sh --args ./build/kissat ./e10.cnf

echo "-------------------------------------"
echo "GDB 会话结束。"