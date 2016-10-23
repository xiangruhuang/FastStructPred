cat $1 | grep 'Best primal obj\|Time' | sed 's/.*Best primal obj = //' | sed 's/\t.*Time = / /' | sed 's/ sec.//' | awk '{print $2 " " $1}' > $2
