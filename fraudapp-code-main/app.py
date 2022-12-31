
import numpy as np
import pandas as pd
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from flask import Flask,render_template,request
import os 
from flask_cors import CORS, cross_origin
from flask import send_from_directory 
nltk.download('punkt')
file_name = 'static/models/modellib.sav'
model = joblib.load(file_name)

app = Flask(__name__,
			static_url_path='',
            static_folder='static',
            template_folder='templates')

CORS(app, support_credentials=True)

@app.route('/')
@app.route('/index',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def index():
    return render_template('index.html')

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict',methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def predict():
    result=""
    dataset = pd.read_excel(os.path.abspath("static/datasets/Output-table-prep.xlsx"))

    titles = dataset.title.str.cat(sep=' ')
    tokens = word_tokenize(titles)
    vocabulary = set(tokens)
    frequency_dist = nltk.FreqDist(tokens)
    repeated_words = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:75]
    repeated_words = [i.upper() for i in repeated_words]
    repeated_words_new = []

    for i in range(len(repeated_words)):
        words = re.sub("[)!&,.''(]",'',repeated_words[i])
        repeated_words_new.append(words)
    repeated_words = list(set(repeated_words_new))    

    repeated_words.sort()
    repeated_words = repeated_words[1:]


    by_state = pd.DataFrame(dataset.groupby(['State','genuine']).count())
    by_state.reset_index(inplace=True)
    by_state = by_state.iloc[:,[0,1,2]]
    state_values = dict(by_state.groupby('State').City.sum())
    prob_by_state = [by_state['City'][i]/state_values[by_state['State'][i]] for i in range(len(by_state['State']))]   
    by_state['prob_by_state'] = prob_by_state
    by_state_no = by_state.loc[by_state['genuine']=='no']
    by_state_yes = by_state.loc[by_state['genuine']=='yes']

    def regex_string(title,benefitted,posted):
        string = ''
        for i in range(len(title)):
            string = re.sub("[!&',.0-9]",'',title.upper())
            string = re.sub(benefitted.upper(),"",string)
            string = re.sub(posted.upper(),"",string)
        return string

    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return np.round((intersection/union),4)


    def per_collected(needed,raised):
        needed = int(needed)
        raised = int(raised)
        return float(raised/needed)

    def statediff(string):
        try:
            return abs(float(by_state_no.loc[by_state_no['State']==string]['prob_by_state'])-float(by_state_yes.loc[by_state_yes['State']==string]['prob_by_state']))
        except:
            return 0

    def make_raised_slab(data):
        data = int(data)
        if data<=20000:
            return 1
        elif data>20000 and data<=50000:
            return 2
        elif data>50000 and data<=100000:
            return 3
        elif data>100000 and data<=250000:
            return 4
        elif data>250000 and data<=500000:
            return 5
        elif data>500000 and data<=750000:
            return 6
        elif data>750000 and data<=1000000:
            return 7
        else:
            return 8

    def make_supporters_slab(data):
        data = int(data)
        if data<=200:
            return 1
        elif data>200 and data<=400:
            return 2
        elif data>400 and data<=600:
            return 3
        elif data>600 and data<=800:
            return 4
        else:
            return 5


    def tax_benefits(string):
        if string.upper()=="YES":
            return 1
        elif string.upper()=="NO":
            return 0


    by_state = pd.DataFrame(dataset.groupby(['State','genuine']).count())
    by_state.reset_index(inplace=True)
    total_state_values = dict(by_state.groupby('State').City.sum())
    by_state_no = by_state.loc[by_state['genuine']=='no']
    by_state_yes = by_state.loc[by_state['genuine']=='yes']
    by_state_no_val = dict(zip(by_state_no['State'],by_state_no['City']))
    by_state_yes_val = dict(zip(by_state_yes['State'],by_state_yes['City']))

    def state(string):
        if string not in by_state_yes_val.keys() and string not in by_state_no_val.keys():
            return 0
        elif string not in by_state_yes_val.keys():
            return float(by_state_no_val[string])/float(total_state_values[string])
        elif string not in by_state_no_val.keys():
            return float(by_state_yes_val[string])/float(total_state_values[string])
        elif by_state_no_val[string]>by_state_yes_val[string]:
            return float(by_state_no_val[string])/float(total_state_values[string])
        else:
            return float(by_state_yes_val[string])/float(total_state_values[string])

    if request.method=="POST":
        title = request.form.get("title")
        posted_by = request.form.get("poname")
        patient = request.form.get("paname")
        state_name   = request.form.get("state")
        tax     = request.form.get("tax")
        supporters_value = request.form.get("supporters")
        needed  = request.form.get("namount")
        raised  = request.form.get('ramount')
        



        new_title = regex_string(title,patient,posted_by)
        jaccard_title = jaccard(repeated_words,new_title)
        state_value = state(state_name)
        state_diff  = statediff(state_name)
        tax_value   = tax_benefits(tax)
        supp_slab   = make_supporters_slab(supporters_value)
        perc_collected_value = per_collected(raised=raised,needed=needed)
        raised_slab = make_raised_slab(raised)         
        needed_slab = make_raised_slab(needed)

        
        
        predict_list = []
        predict_list.append(supporters_value)
        predict_list.append(tax_value)
        predict_list.append(raised)
        predict_list.append(needed)
        predict_list.append(state_value)
        predict_list.append(state_diff)
        predict_list.append(jaccard_title)
        predict_list.append(perc_collected_value)
        predict_list.append(supp_slab)
        predict_list.append(raised_slab)
        predict_list.append(needed_slab)
        
        

        X = pd.DataFrame(np.reshape(predict_list,(1,11)),columns=['supporters', 'tax benifits',  'raised_amount',
       'needed_amount', 'State1','statediff', 'jaccard_title', 'perc_collected',
    'supporters_slab','raised_slab','needed_slab'])
      

        
        result = model.predict_proba(X)
        perc1 = result.tolist()[0][0]
        perc2 = result.tolist()[0][1]
        if perc2>perc1:
            return render_template('genuine.html',variable=np.round(perc2*100))
        else:
            return render_template('notgenuine.html',variable=np.round(perc1*100))    



if __name__=="__main__":
    app.run(debug=True)

