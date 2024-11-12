#!/bin/bash
# sudo apt install clang-format
find . -regex '.*\.\(cpp\|hpp\|cu\|c\|h\|cc\)' -exec clang-format -style=file -i {} \;
