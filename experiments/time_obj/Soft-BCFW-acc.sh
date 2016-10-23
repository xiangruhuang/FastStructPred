cat $1 | grep 'best_decoded' | sed 's/.* best_decoded=//' | sed 's/,.*time=/ /' | awk '{print $2 " " $1}' > $2
