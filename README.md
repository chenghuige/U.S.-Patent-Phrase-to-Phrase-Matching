# training code  
cd projects/kaggle/usp    
# I used 7 models * 6 = 42 total models as below  
python ./main.py 
python ./maing.py --hug=patent  
python ./main.py --hug=electra-squad  
python ./maing.py --hug=simcse-patent  
python ./main.py  --context_key=sector  
python ./maing.py --hug=patent  --context_key=sector  
python ./main.py --hug=electra-squad  --context_key=sector  
# 5 folds --fold=1 --fold=2 --fold=3 --fold=4
# full_train --online  

