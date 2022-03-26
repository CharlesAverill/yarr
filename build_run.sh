echo "---CLANG-FORMAT----"
clang-format style=file -i $(find src -name "*.cu") $(find include -name "*.cuh")
echo "-----CMAKE---------"
cd ~/CProjects/yarr/build
if cmake .. ; then
    echo "-----MAKE----------"
    if make -j$(nproc) ; then
        cd ../bin
        echo "-----EXECUTING-----"
        rm "$1"
        rm "$1.mp4"
        if ./yarr $1 ; then
            echo "-----CONVERTING----"
            ffmpeg -hide_banner -loglevel error -y -i "$1" "$1.mp4" > /dev/null
            echo "Done"
        fi
    else
        echo "-----MAKE FAILURE-----"
    fi
else
    echo "-----CMAKE FAILURE-----"
fi
