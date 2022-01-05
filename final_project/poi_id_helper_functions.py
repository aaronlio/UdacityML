def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    #error = actual - predicted
    cleaned_data = []
    error = sorted(abs(net_worths-predictions))[0:81]
    max_error = max(error)
    for i in range(max(len(net_worths), len(predictions))):
        if (abs(net_worths[i]-predictions[i]) <= max_error):
            cleaned_data.append((ages[i], net_worths[i], net_worths[i]-predictions[i]))
    

    ### your code goes here
    return cleaned_data