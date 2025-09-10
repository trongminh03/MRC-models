# Machine Reading Comprehension Models

## How to run the code  
For training: please run the train.py script.   
For evaluation: please run the test.py script.

Required params (for both train.py and test.py):  
--path: Path to the ViSQA dataset  
--type: Type of model (eg. bartpho, xlm-r, phobert, etc). Default is auto   
--model: the path to pre-trained model or slug as shown in the Hugging Face website (eg. xlm-r-base)   
--output_path: Path to the output directory   
--is_test: Pass this param in the test.py if you want to run the evaluation on the test set. If none, the code will run evaluation on the development (dev) set.   

