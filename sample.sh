for i in {0..99}
    do
        echo "Sample #: $i"
        python sample.py ./configs/sample.yml --data_id $i
    done
