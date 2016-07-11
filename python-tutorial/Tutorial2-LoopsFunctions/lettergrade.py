#if statment example
grade = int (raw_input("Enter precentage to calculate letter grade? "))
if grade >= 90 and grade <= 100:
  lettergrade = 'A'
elif grade >= 80 and grade < 90:
  lettergrade = 'B'
elif grade >= 70 and grade < 80:
  lettergrade = 'C'
elif grade >= 60 and grade < 70:
  lettergrade = 'D'
elif grade < 60:
  lettergrade = 'F'
else:
  lettergrade = 'Not Valid'
  
print ("Your Letter Grade = ", lettergrade)  