echo "-----CMAKE---------"
cd ~/CProjects/yarr/build
if cmake .. ; then
    echo "-----MAKE----------"
    if make ; then
        cd ../bin
        echo "-----EXECUTING-----"
        rm $1
        if ./yarr $1 ; then
            eog $1
        fi
    else
        echo "-----MAKE FAILURE-----"
    fi
else
    echo "-----CMAKE FAILURE-----"
fi
