#!/usr/bin/python


def calc_error(predicted_worth, actual_worth):
    return predicted_worth - actual_worth


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = [(ages[i], net_worths[i], calc_error(predictions[i], net_worths[i])) for i in range(len(ages))]

    return sorted(cleaned_data, key=lambda tup: tup[2])[:81]

