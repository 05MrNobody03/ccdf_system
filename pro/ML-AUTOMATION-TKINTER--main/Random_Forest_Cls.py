# import tkinter as tk
# from tkinter import *
# from tkinter import ttk
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score,r2_score
# from sklearn.impute import SimpleImputer
# from tkinter.filedialog import askopenfilename
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.model_selection import GridSearchCV

# feature_col = []
# target_col = []

# root = Tk()
# root.title('Machine Learning GUI')
# root.geometry('800x750')
# root.title("Best Regressor")


# # def selectall():
# #     box1.select_set(0, END)


# def data():
#     global filename
#     filename = askopenfilename(initialdir=r'C:\Users\surya\Desktop\CDAC Noida\ML\Files', title="Select file")
#     e1.insert(0, filename)
#     e1.config(text=filename)

#     global file
#     file = pd.read_csv(filename)
#     for i in file.columns:
#         box1.insert(END, i)

#     for i in file.columns:
#         if type(file[i][0]) == np.float64 :
#             file[i].fillna(file[i].mean(), inplace=True)
#         elif type(file[i][0]) == np.int64 :
#             file[i].fillna(file[i].median(), inplace=True)
#         elif type(file[i][0]) == type(""):
#             imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#             s = imp.fit_transform(file[i].values.reshape(-1, 1))
#             file[i] = s

#     colss = file.columns
#     global X_Axis
#     X_Axis = StringVar()
#     X_Axis.set('X-axis')
#     choose = ttk.Combobox(root, width=22, textvariable=X_Axis)
#     choose['values'] = (tuple(colss))
#     choose.place(x=400, y=20)

#     global Y_Axis
#     Y_Axis = StringVar()
#     Y_Axis.set('Y-axis')
#     choose = ttk.Combobox(root, width=22, textvariable=Y_Axis)

#     choose['values'] = (tuple(colss))
#     choose.place(x=400, y=40)
#     global graphtype
#     graphtype = StringVar()
#     graphtype.set('Graph')
#     choose = ttk.Combobox(root, width=22, textvariable=graphtype)
#     choose['values'] = ('scatter', 'line', 'bar', 'hist', 'corr', 'pie')
#     choose.place(x=400, y=60)

# def getx():
#     x_v = []
#     s = box1.curselection()
#     global feature_col
#     for i in s:
#         if i not in feature_col:
#             feature_col.append((file.columns)[i])
#             x_v = feature_col
#     for i in x_v:
#         box2.insert(END, i)

# def gety():
#     y_v = []
#     global target_col
#     s = box1.curselection()
#     for j in s:
#         if j not in target_col:
#             target_col.append((file.columns)[j])
#             y_v = target_col

#     for i in y_v:
#         box3.insert(END, i)

# def plot():
#     fig = Figure(figsize=(6, 6), dpi=70)
#     global X_Axis
#     global Y_Axis
#     global graphtype
#     u = graphtype.get()

#     if u == 'scatter':
#         plot1 = fig.add_subplot(111)
#         plt.scatter(file[X_Axis.get()], file[Y_Axis.get()])
#         plt.xlabel(X_Axis.get())
#         plt.ylabel(Y_Axis.get())
#         plt.show()

#     if u == 'line':
#         plot1 = fig.add_subplot(111)
#         plt.plot(file[X_Axis.get()], file[Y_Axis.get()])
#         plt.xlabel(X_Axis.get())
#         plt.ylabel(Y_Axis.get())
#         plt.show()

#     if u == 'bar':
#         plot1 = fig.add_subplot(111)
#         plt.bar(file[X_Axis.get()], file[Y_Axis.get()])
#         plt.xlabel(X_Axis.get())
#         plt.ylabel(Y_Axis.get())
#         plt.show()

#     if u == 'hist':
#         plot1 = fig.add_subplot(111)
#         plt.hist(file[X_Axis.get()])
#         plt.xlabel(X_Axis.get())
#         plt.ylabel(X_Axis.get())
#         plt.show()

#     if u == 'corr':
#         plot1 = fig.add_subplot(111)
#         sns.heatmap(file.corr())
#         plt.show()

#     if u == 'pie':
#         plot1 = fig.add_subplot(111)
#         plt.pie(file[Y_Axis.get()].value_counts(), labels=file[Y_Axis.get()].unique())
#         plt.show()



# def model():
    
#     x = file[feature_col]
#     y = file[target_col]
    
#     x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=float(splits.get()))


#     parameters = {"n_estimators": list(range(10,50,5))}


#     tree = DecisionTreeRegressor()
#     boost = AdaBoostRegressor(base_estimator = tree)
#     model = GridSearchCV(boost, param_grid=parameters, scoring = 'r2')
#     model.fit(x_train,np.ravel(y_train))
#     model=model.best_estimator_
#     model.fit(x_train,np.ravel(y_train))
            
#     x_dummies = s.get().split(",")
#     x_tests = []
    
#     for i in x_dummies:
#         x_tests.append(float(i))
        
#     y_pred = model.predict([x_tests])
    
   
#     train_accuracy = r2_score(np.ravel(y_train), model.predict(x_train))
#     test_accuracy = r2_score(np.ravel(y_test), model.predict(x_test))

#     Label(root, text=str(y_pred), font=('Helvetica', 10, 'bold'), bg="light blue", relief="solid").place(x=400,
#                                                                                                          y=450)
#     Label(root, text=f'Train accuracy : {train_accuracy}', font=('Helvetica', 10, 'bold'), bg="light blue",
#           relief="solid").place(x=20, y=550)
#     Label(root, text=f'Test accuracy  : {test_accuracy}', font=('Helvetica', 10, 'bold'), bg="light blue",
#           relief="solid").place(x=20, y=600)

#     if abs(train_accuracy - test_accuracy) > 0.1 or (train_accuracy < .6 and test_accuracy < .6):
#         Label(root, text='BAD  MODEL', font=('Helvetica', 10, 'bold'), bg="red", relief="solid").place(x=300,
#                                                                                                               y=550)

#         if (train_accuracy > test_accuracy) and abs(train_accuracy - test_accuracy) > 0.1:
#             Label(root, text='OVERFIT', font=('Helvetica', 10, 'bold'), bg="red", relief="solid").place(
#                 x=400,
#                 y=550)

#         if train_accuracy < 0.6 and test_accuracy < 0.6:
#             Label(root, text='UNDERFIT', font=('Helvetica', 10, 'bold'), bg="red", relief="solid").place(
#                 x=400,
#                 y=550)

#     elif (train_accuracy < test_accuracy) and abs(train_accuracy - test_accuracy) > 0.1:
#         Label(root, text='GOOD MODEL', font=('Helvetica', 10, 'bold'), bg="green", relief="solid").place(x=300,
#                                                                                                               y=550)

#     elif train_accuracy > 0.85 and test_accuracy > 0.85:
#         Label(root, text='VERY GOOD MODEL', font=('Helvetica', 10, 'bold'), bg="green", relief="solid").place(
#             x=300, y=550)

#     return train_accuracy, test_accuracy, x_tests, y_pred

# def files():
#     with open(r"C:\Users\musan\Desktop\model summary", "w", encoding="utf-8") as file:
#         file.write("You have use Decision Tree classifier model \n")
#         file.write("\n")
#         file.write(f"The columns used for features are {feature_col} and the targetted columns are {target_col}\n")
#         file.write("\n")
#         file.write(
#             f"The train accuracy of the model is {model()[0]} and The test accuracy of the model is are and {model()[1]}\n")
#         file.write("\n")

#         if model()[0] > .9 and model()[1] > .9:
#             file.write("The model is Excellent")
#         if abs(model()[0] - model()[1]) > .1 and (model()[0] < .6 and model()[1] < .6):
#             file.write("The model is an Underfitted one")
#         if model()[0] > .6 and model()[1] > .6 and abs(model()[0] - model()[1]) > .1:
#             file.write(f"The model is an Overfitted one")
#         file.write("\n")
#         file.write(f"The User inputs were {model()[2]} and the predicted output was {model()[3]}\n")

# listbox = Listbox(root, selectmode="multiple")
# listbox.pack


# Label(root, font="System", text="split_size").place(x=20, y=300)
# splits = StringVar()
# choose = ttk.Combobox(root, width=30, textvariable=splits)
# choose['values'] = ('0.2', '0.25', '0.3')
# choose.place(x=150, y=300)

# s = StringVar()
# Entry(root,text=s,width=30).place(x=250,y=450)
# Label(root,font="System",text='Inputs separated by commas').place(x=20,y=450)

# l1 = Label(root, text='Select Data File')
# l1.grid(row=0, column=0)
# e1 = Entry(root, text='')
# e1.grid(row=0, column=1)
# Button(root, text='open', command=data,activeforeground="white",activebackground="black").grid(row=0, column=2)

# box1 = Listbox(root, selectmode='multiple')
# box1.grid(row=10, column=0)

# #Button(root, text='Select all', command=selectall,activeforeground="white",activebackground="black").grid(row=13, column=1)

# box2 = Listbox(root)
# box2.grid(row=10, column=1)
# Button(root, text='Select X', command=getx,activeforeground="white",activebackground="black").grid(row=13, column=1)

# box3 = Listbox(root)
# box3.grid(row=10, column=2)
# Button(root, text='Select Y', command=gety,activeforeground="white",activebackground="black").grid(row=13, column=2)

# Button(root, text="Plot", command=plot,activeforeground="white",activebackground="black").place(x=600, y=50)

# Button(root, text="predict", command=model,activeforeground="white",activebackground="black").grid(row=13, column=3)

# Button(root, text="Summary", command=files,activeforeground="white",activebackground="black").place(x=450, y=190)
# root.mainloop()



































import tkinter as tk
from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,r2_score
from sklearn.impute import SimpleImputer
from tkinter.filedialog import askopenfilename
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV


from tkinter.filedialog import asksaveasfilename

feature_col = []
target_col = []

root = Tk()
root.title('Machine Learning GUI')
root.geometry('800x750')
root.title("Random Forest Classifier")

def takefileinput():
    pass

def selectall():
    pass
    box1.select_set(0, END)


def data():
    global filename
    filename = askopenfilename(initialdir=r'C:\Users\surya\Desktop\CDAC Noida\ML\Files', title="Select file")
    e1.insert(0, filename)
    e1.config(text=filename)

    global file
    file = pd.read_csv(filename)
    for i in file.columns:
        box1.insert(END, i)

    for i in file.columns:
        if type(file[i][0]) == np.float64 :
            file[i].fillna(file[i].mean(), inplace=True)
        elif type(file[i][0]) == np.int64 :
            file[i].fillna(file[i].median(), inplace=True)
        elif type(file[i][0]) == type(""):
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            s = imp.fit_transform(file[i].values.reshape(-1, 1))
            file[i] = s

    colss = file.columns
    global X_Axis
    X_Axis = StringVar()
    X_Axis.set('X-axis')
    choose = ttk.Combobox(root, width=22, textvariable=X_Axis)
    choose['values'] = (tuple(colss))
    choose.place(x=400, y=20)

    global Y_Axis
    Y_Axis = StringVar()
    Y_Axis.set('Y-axis')
    choose = ttk.Combobox(root, width=22, textvariable=Y_Axis)

    choose['values'] = (tuple(colss))
    choose.place(x=400, y=40)
    global graphtype
    graphtype = StringVar()
    graphtype.set('Graph')
    choose = ttk.Combobox(root, width=22, textvariable=graphtype)
    choose['values'] = ('scatter', 'line', 'bar', 'hist', 'corr', 'pie')
    choose.place(x=400, y=60)

def getx():
    x_v = []
    s = box1.curselection()
    global feature_col
    for i in s:
        if i not in feature_col:
            feature_col.append((file.columns)[i])
            x_v = feature_col
    for i in x_v:
        box2.insert(END, i)

def gety():
    y_v = []
    global target_col
    s = box1.curselection()
    for j in s:
        if j not in target_col:
            target_col.append((file.columns)[j])
            y_v = target_col

    for i in y_v:
        box3.insert(END, i)

def plot():
    fig = Figure(figsize=(6, 6), dpi=70)
    global X_Axis
    global Y_Axis
    global graphtype
    u = graphtype.get()
    df=pd.DataFrame(file)
    print(df.head())
    if u == 'scatter':
        plot1 = fig.add_subplot(111)
        groups = df.groupby('Class')
        print("groups: \n\n\n",groups.head(),"\n\n\n")
        for name, group in groups:
            plt.plot(group[X_Axis.get()], group[Y_Axis.get()], marker='o', linestyle='', markersize=6, label=name)

        # plt.scatter(file[X_Axis.get()], file[Y_Axis.get()])
        plt.legend()
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u == 'line':
        plot1 = fig.add_subplot(111)
        plt.plot(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u == 'bar':
        plot1 = fig.add_subplot(111)
        plt.bar(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u == 'hist':
        plot1 = fig.add_subplot(111)
        plt.hist(file[X_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(X_Axis.get())
        plt.show()

    if u == 'corr':
        plot1 = fig.add_subplot(111)
        sns.heatmap(file.corr())
        plt.show()

    if u == 'pie':
        plot1 = fig.add_subplot(111)
        plt.pie(file[Y_Axis.get()].value_counts(), labels=file[Y_Axis.get()].unique())
        plt.show()



def opentestfile():
    global testfile
    testfile = askopenfilename(initialdir=r'C:\Users\surya\Desktop\CDAC Noida\ML\Files', title="Select file")
    e2.insert(0, testfile)
    e2.config(text=testfile)

    global test
    test = pd.read_csv(testfile)
    test=test.drop("Class",axis=1)
    test_len=len(test)
    print(test_len)
    test1=test.iloc[0:,0:30]
    DF=pd.DataFrame(test1)
    pred_values=[]
    
    for p in range(0,test_len):

        r1=test.iloc[p, 0:30 ]
    
        # print(r1)
        # print("type of r1 is : ",type(r1))
        l=[]
        temp_l=[]
        for i in r1:
            l.append(float(i))
        pred=model.predict([l])
        pred_values.append(int(pred))

    
    # l1 = l
    # l2=[feature_col]

    # dic={}
    # dic1={}
    # DF=pd.DataFrame()
    # for z in range(5):
        
    #     for (i,j) in zip(l2,l1):
    #         dic[str(i)]=float(j)
    #     df=pd.DataFrame(dic,index=[z])
        
    #     DF = DF.append(df, ignore_index=True)

    
    DF.insert(2, "Predicted Values", pred_values, True)
    filename2 = asksaveasfilename(filetype=[('CSV files', '*.csv')])
    if filename2:
        #df.to_csv(filename, header=False, index=False)
        DF.to_csv(filename2, index=False)

            
    print(DF)


    #Label(root, text=str(pred), font=('Helvetica', 10, 'bold'), bg="light blue", relief="solid").place(x=800,y=550)
   
    pass


def model():
    
    x = file[feature_col]
    y = file[target_col]
    
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=float(splits.get()))


    parameters = {"n_estimators": list(range(10,50,5))}


    tree = DecisionTreeRegressor()
    boost = AdaBoostRegressor(base_estimator = tree)
    global model 
    model = GridSearchCV(boost, param_grid=parameters, scoring = 'r2')
    model.fit(x_train,np.ravel(y_train))
    model=model.best_estimator_
    model.fit(x_train,np.ravel(y_train))
            
    x_dummies = s.get().split("\t")
    x_tests = []
    
    for i in x_dummies:
        x_tests.append(float(i))
    print("type of x_tests is:",type([x_tests]))                     ##
    y_pred = model.predict([x_tests])
    
   
    train_accuracy = r2_score(np.ravel(y_train), model.predict(x_train))
    test_accuracy = r2_score(np.ravel(y_test), model.predict(x_test))

    Label(root, text=str(y_pred), font=('Helvetica', 10, 'bold'), bg="light blue", relief="solid").place(x=450,
                                                                                                         y=450)
    # Label(root, text=f'Train accuracy : {train_accuracy}', font=('Helvetica', 10, 'bold'), bg="light blue",
    #       relief="solid").place(x=20, y=550)
    # Label(root, text=f'Test accuracy  : {test_accuracy}', font=('Helvetica', 10, 'bold'), bg="light blue",
    #       relief="solid").place(x=20, y=600)

    # if abs(train_accuracy - test_accuracy) > 0.1 or (train_accuracy < .6 and test_accuracy < .6):
    #     Label(root, text='BAD  MODEL', font=('Helvetica', 10, 'bold'), bg="red", relief="solid").place(x=300,
    #                                                                                                           y=550)

    #     if (train_accuracy > test_accuracy) and abs(train_accuracy - test_accuracy) > 0.1:
    #         Label(root, text='OVERFIT', font=('Helvetica', 10, 'bold'), bg="red", relief="solid").place(
    #             x=400,
    #             y=550)

    #     if train_accuracy < 0.6 and test_accuracy < 0.6:
    #         Label(root, text='UNDERFIT', font=('Helvetica', 10, 'bold'), bg="red", relief="solid").place(
    #             x=400,
    #             y=550)

    # elif (train_accuracy < test_accuracy) and abs(train_accuracy - test_accuracy) > 0.1:
    #     Label(root, text='GOOD MODEL', font=('Helvetica', 10, 'bold'), bg="green", relief="solid").place(x=300,
    #                                                                                                           y=550)

    # elif train_accuracy > 0.85 and test_accuracy > 0.85:
    #     Label(root, text='VERY GOOD MODEL', font=('Helvetica', 10, 'bold'), bg="green", relief="solid").place(
    #         x=300, y=550)

    return train_accuracy, test_accuracy, x_tests, y_pred

def files():
    with open(r"C:\Users\musan\Desktop\model summary", "w", encoding="utf-8") as file:
        file.write("You have use Decision Tree classifier model \n")
        file.write("\n")
        file.write(f"The columns used for features are {feature_col} and the targetted columns are {target_col}\n")
        file.write("\n")
        file.write(
            f"The train accuracy of the model is {model()[0]} and The test accuracy of the model is are and {model()[1]}\n")
        file.write("\n")

        if model()[0] > .9 and model()[1] > .9:
            file.write("The model is Excellent")
        if abs(model()[0] - model()[1]) > .1 and (model()[0] < .6 and model()[1] < .6):
            file.write("The model is an Underfitted one")
        if model()[0] > .6 and model()[1] > .6 and abs(model()[0] - model()[1]) > .1:
            file.write(f"The model is an Overfitted one")
        file.write("\n")
        file.write(f"The User inputs were {model()[2]} and the predicted output was {model()[3]}\n")

listbox = Listbox(root, selectmode="multiple")
listbox.pack


Label(root, font="System", text="split_size").place(x=20, y=300)
splits = StringVar()
choose = ttk.Combobox(root, width=30, textvariable=splits)
choose['values'] = ('0.2', '0.25', '0.3')
choose.place(x=150, y=300)

s = StringVar()
Entry(root,text=s,width=30).place(x=250,y=450)
Label(root,font="System",text='Inputs separated by commas').place(x=20,y=450)

l1 = Label(root, text='Select Data File')
l1.grid(row=0, column=0)
e1 = Entry(root, text='')
e1.grid(row=0, column=1)
Button(root, text='open', command=data,activeforeground="white",activebackground="black").grid(row=0, column=2)

box1 = Listbox(root, selectmode='multiple')
box1.grid(row=10, column=0)

#Button(root, text='Select all', command=selectall,activeforeground="white",activebackground="black").grid(row=13, column=1)

box2 = Listbox(root)
box2.grid(row=10, column=1)
Button(root, text='Select X', command=getx,activeforeground="white",activebackground="black").grid(row=13, column=1)

box3 = Listbox(root)
box3.grid(row=10, column=2)
Button(root, text='Select Y', command=gety,activeforeground="white",activebackground="black").grid(row=13, column=2)

Button(root, text="Plot", command=plot,activeforeground="white",activebackground="black").place(x=600, y=50)

Button(root, text="predict", command=model,activeforeground="white",activebackground="black").grid(row=13, column=3)

Button(root, text="Summary", command=files,activeforeground="white",activebackground="black").place(x=450, y=190)



Label(root,font="System",text='Inputs separated by commas').place(x=20,y=450)


s1 = StringVar()
# e2 = Entry(root,text=s1,width=30).place(x=250,y=480)


l2 = Label(root, text='Select Data File')
l2.grid(row=0, column=0)
e2 = Entry(root, text='')
e2.place(x=250,y=480)
# e2.place(x=250,y=480)
Label(root,font="System",text='Inputs test data here').place(x=20,y=480)
Button(root, text="open1", command=opentestfile ,activeforeground="white",activebackground="black").place(x=450, y=480)



root.mainloop()

