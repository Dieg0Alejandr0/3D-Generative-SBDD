echo "WE BEGIN!"
cat remainder.txt | while read line 
do
   echo "Sample #: $line"
   python sample.py ./configs/sample.yml --data_id $line
done
