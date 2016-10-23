cat $1 | grep 'best_value=' | sed 's/^best_value=//' | sed 's/,.* time=/ /' | awk '{print $2 " " $1 }' > $2
