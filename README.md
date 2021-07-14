![Image](https://user-images.githubusercontent.com/80604404/125493756-95f9b336-9894-472f-88c6-30399488c910.png)


### PROJECT DESCRIPTION


Finance companies deal with many types of home loans. The customer first applies for home loan, after that the finance company validates the customer eligibility for loan. These companies are looking to automate the loan eligibility process (real time).

I used the classification methodology that predicts the loan status based on customer information provided while completing the online application. The details included are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. As my target is categorical (loan_status : approved or not approved), I utilized a binary classification predictive model. In order to create the model, I cleaned my data and decided how to handle missing values, created new columns, scaled my data, and explored my data. My best model (KNN) performed better than my baseline.



### GOALS 

- find drivers for loan_status
- Constuct a ML  model that accurately predicts loan status.


### DATA DICTIONARY
---
| Attribute | Definition | Data Type | Values|
| ----- | ----- | ----- | ----- |
|loan_id | Unique Loan ID | Object| alphanumeric|
|gender | gender od each applicant| int64 | 0= Female, 1 = Male |
|married| married applicant | int64 | 0= No, 1 + Yes |
|dependents | how many dependents | float64 | 0= none, 1 = 1 dependent, 2 = 2 dependents, 3 = 3 or more |
|self_employed | self employed | int64 | 0= No, 1 + Yes |
|applicantincome| applicant income| int64 |  dollars|
|coapplicantincome | coapplicant income | float64 | dollars
|loanamount | loan amount| float64 | dollars |
|loan_amount_term | Term of loan| float64 | months|
|credit_history | credit history meets guidelines | uint8 | 0= No, 1 + Yes |
|total_income | the sum of applicant and coapplicant income| float64| dollars|
|education_graduate | graduated applicant | uint8 |  0= No, 1 + Yes |
|property_area_rural | rural area |uint8 |  0= No, 1 + Yes |
|property_area_semiurban | semiurban area |uint8 |  0= No, 1 + Yes |
|property_area_urban | urban area |uint8 |  0= No, 1 + Yes |
|has_coapplicant | loan has a coapplicant | int64 | 0= No, 1 + Yes |
|income_portion_dependents | totalincome divided by dependents +1 | float64 | dollars|
|**loan_status** | loan status| int64 |0= No, 1 + Yes | 
___



### PROJECT PLANNIG
[Trello](https://trello.com/b/U4a2HdTV/loan-application-project)

- Acquire data from [Kaggle](https://www.kaggle.com/vipin20/loan-application-data) .
- Clean and prepare data for preparation step. Create a function to automate the process. The function is saved in a prepare.py module.
- Explore my data using different visualizations , define two hypotheses, set an alpha, run the statistical tests needed,document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train  models.
- Evaluate models on train and validate datasets.
- Choose the model that performs the best.
- Evaluate the best model (only one) on the test dataset




### INITIAL IDEAS/ HYPOTHESES STATED
- 𝐻𝑜 : Rate of loan_status approval is not dependent on loan_amount_term.
- 𝐻𝑎 : Rate of loan_status approval is dependent on loan_amount_term

### INSTRUCTIONS FOR RECREATING PROJECT

- [x] Read this README.md
- [ ] Download the csv file from [Kaggle](https://www.kaggle.com/vipin20/loan-application-data) and save it in the same directory where  you are going to recreate the poject.
- [ ] Download the aquire.py, prepare.py, model.py, explore.py and  final_notebook_project.ipynb into your working directory
- [ ] Run the final_notebook_project.ipynb notebook


### DELIVER:
- A Jupyter Notebook Report showing process and analysis with goal of finding drivers .
- A README.md file containing project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), and key findings and takeaways from your project.
- Individual modules, .py files, that hold your functions to acquire and prepare your data.
