cat $1 | grep 'Decoded' | sed 's/.* Decoded=//' | sed 's/ .*Elapsed Time=/ /' | awk '{print $2 " " $1}' > $2
