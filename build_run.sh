echo "---CLANG-FORMAT----"
clang-format style=file -i src/* src/renderobjects/* src/utils/* include/* include/renderobjects/* include/utils/*
echo "-----CMAKE---------"
cd ~/CProjects/yarr/build
if cmake .. ; then
    echo "-----MAKE----------"
    if make -j$(nproc) ; then
        cd ../bin
        echo "-----EXECUTING-----"
        rm $1
        rm "$1.png"
        if ./yarr $1 ; then
            convert $1 "$1.png"
            # eog "$1.png"
        fi
    else
        echo "-----MAKE FAILURE-----"
    fi
else
    echo "-----CMAKE FAILURE-----"
fi
