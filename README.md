### PROJECT DESCRIPTION

Finance companies deals with some kinds of home loans. Customer first applies for home loan and after that company validates the customer eligibility for loan. Using the data from Kaggel 

Companies want to automate the loan eligibility process (real time).

I created a classification model that predicts loan status based on customer details provided while filling online application form.  These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others

### GOALS 

- find drivers for loan approval
- Constuct a ML a ML  model that accurately predicts loan approval (in this case is a classification model because the target is categorical)


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
|credit_history | credit history meets guidelines |  0= No, 1 + Yes |
|total_income | the sum of applicant and coapplicant income| float64| dollars|
|education_graduate | graduated applicant | uint8 |  0= No, 1 + Yes |
|property_area_rural | rural area |uint8 |  0= No, 1 + Yes |
|property_area_semiurban | semiurban area |uint8 |  0= No, 1 + Yes |
|property_area_urban | urban area |uint8 |  0= No, 1 + Yes |
|has_coapplicant | loan has a coapplicant | int64 | 0= No, 1 + Yes |
|income_portion_dependents | totalincome divided by dependents +1 | float64 | dollars|
|**loan_status** | loan approved| int64 |0= No, 1 + Yes | 
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
- 𝐻𝑜 : 
- 𝐻𝑎 : 

### INSTRUCTIONS FOR RECREATING PROJECT

- [x] Read this README.md
- [ ] Download the csv file from [Kaggle](https://www.kaggle.com/vipin20/loan-application-data) and save it in the same directory where are you goint to recreate the poject.
- [ ] Download the aquire.py, prepare.py, model.py, explore.py and  final_notebook_project.ipynb into your working directory
- [ ] Run the final_notebook_project.ipynb notebook


### DELIVER:
- A Jupyter Notebook Report showing process and analysis with goal of finding drivers .
- A README.md file containing project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), and key findings and takeaways from your project.
- Individual modules, .py files, that hold your functions to acquire and prepare your data.
