mkdir datasets/$1
mkdir datasets/$1/train
mkdir datasets/$1/val
mkdir datasets/$1/test
cp $1.txt datasets/$1/train/$1.txt
cp $1.txt datasets/$1/test/$1.txt
cp $1.txt datasets/$1/val/$1.txt
mv $1.txt controlled_setup/$1.txt