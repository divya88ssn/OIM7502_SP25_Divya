# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### Python Programming Concepts

# #### 1. Create a variable to store the value 500 for sales
sales = 500

# #### 2. Create a variable to store several values:  
# * sales: 500, 475, 625
# * integers 1 - 9
sales_val_list = [500, 475, 625]

# #### 3. Create a variable to store the names and email addresses for a number of customers
# * John, john@some.com; Ann, ann@some.com

person1 = dict(name="John", email="john@some.com")
person2 = dict(name="Ann", email="ann@some.com")

customers_list = [person1, person2]

# #### 4. Use Python to generate a random value based on each of following:
# 1. between 0 and 1
# 2. the standard normal distribution (rounded to two decimal places)
# 3. a value between 1 and 10
# 4. either H or T

value1 = random.random()
value2 = round(np.random.randn(), 2)
value3 = random.randint(1, 10)
value4 = random.choice(["H", "T"])

# #### 5. Use Python to determine your current working directory
current_directory = os.getcwd()

# #### 6. Use Python to determine how many days until New Year's Day



# #### 7. Write a short program that displays the future value of 1,000 earning 5%  at the end of each year for the next 5 years



# #### 8. Write a short program that prompts a user to enter a stock symbol and press enter. Add each symbol entered to a variable. The program should run until the user presses the enter key without entering anything (empty string) 



# #### 9. Write the code necessary to calculate the sum of the square differences of a group of values : 
# 58, 32, 37, 41, 36, 36, 54, 37, 25, 53
# ##### $\Sigma $(${x}$ - $\bar{x}$)$^2$



# #### 10. Opening, reading and writing files



# #### 11. Create a function the will simulate rolling two die and return the value of each and and the sum 



# #### 12. Write the code needed to evaluate whether a value in a group of values is negative or postive and prints an approriate message, i.e. postive, negative 
# -1.59, 2.36, 1.69, 1.13, -0.91, 1.48, -0.34, 1.31, -0.74, 0.2





# #### 14.  Complete the following:
# 1. Write a program that generates 500 random integers from 1 to n, for example, the first integer will be 1 the second will be 1 or 2, the fifth between 1 and 5, and so on.
# 2. The program should write each integer to a file 
# 3. Once all integers are written close the file
#
# ##### Part 2
# 1. Open and read the file into a list
# 2. Create a line plot of the data












