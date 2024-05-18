Business needs

    This project aims to develop an automated system to detect and classify types of bullying in tweets. The aim is to create an effective tool for identifying bullying. This will help social platforms to respond quickly to bullying incidents, providing a safer environment for users.
    
Requirements

    python 3.12

    numpy==1.26.4
    pandas==1.5.3
    sklearn==1.4.2
    demoji==0.2.1
    nltk==3.8.1
    imblearn=0.12.2

Running:

    To run the demo, execute:
        python ./pipeline/predict.py

    After running the script in the root folder will be generated <prediction_results.csv>
    The file has 'sentiment' column with the result value.

    The input is expected  csv file in the same folder <data/> with a name <new_data.csv>. The file should have only text column.

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain text column and target sentiment label column.
    After running the script the <finalized_model.saw>, <clf.saw> and <tf_transformer.saw> will be created model folder.
    Run the training script:
        python ./pipeline/train.py

    The model accuracy is 86%
