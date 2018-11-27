EPOCHS=50

python inn.py  --epochs ${EPOCHS} --act linear & 
python inn.py  --epochs ${EPOCHS} --act sub &
python inn.py  --epochs ${EPOCHS} --act supra &
python inn.py  --epochs ${EPOCHS} --act mixed &



