# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:37:29 2024

@author: Campbell
"""

import os
import sys
import json
import re

import numpy as np
import pandas as pd

from datetime import date
from dateutil.relativedelta import relativedelta

# Code variables
specimen_statuses = ['Available', 'Shipped']
blood_sample_types = ['10^7 PBMC', 'Plasma', 'Whole Blood']

def return_REDCap_data(redcap_file_string):
    """ Code pulls data from REDCap via the API, restricts to needed columns,
        returns dataframe with participant information """

    # Variables
    columns_to_keep = ['record_id', 'redcap_event_name', 'demo_uk_mrn',
                        'visit_date', 'visit_liver_procured',
                        'visit_fat_procured', 'visit_blood_procured']
   
    # Code
    
    # Load the file as a dataframe, keeping leading zeros on MRN
    d = pd.read_csv(redcap_file_string, converters={'demo_uk_mrn': str})

    # Restrict to needed columns
    d = d[columns_to_keep]
    
    # Fill empty entries with NaN to then forward fill ukmrn
    d['demo_uk_mrn'] = d['demo_uk_mrn'].replace('', np.nan)
    
    # Forward fill the MRN
    d['demo_uk_mrn'] = d['demo_uk_mrn'].infer_objects(copy=False).ffill()
    
    # Convert mrn to string
    d['demo_uk_mrn'] = d['demo_uk_mrn'].astype(str)
    
    # Convert dates
    d = convert_series_to_datetimes(d)

    # Return    
    return d

def return_OnCore_data(oncore_file_string):
    """ Loads full report from OnCore, restricts to needed columns,
        returns dataframe with inventory information """
    
    # Variables
    columns_to_keep = ['Patient ID', 'Collection Date', 'Specimen No.',
                       'Specimen Status', 'Specimen Type', 'Body Site']
    
    # Code
    
    # Read in data, keeping leading zeros on MRN
    d = pd.read_csv(oncore_file_string, converters={'Patient ID': str})
     
    # Tidy mrns
    for i in range(len(d)):
        d['Patient ID'].iat[i] = tidy_mrn(d['Patient ID'].iloc[i])
 
    # Restrict to needed columns
    d = d[columns_to_keep]
 
    # Convert dates
    d = convert_series_to_datetimes(d)
    
    
    print('len d: %i' % len(d))
    
    # Drop the deaccesioned samples
    bi = d[d['Specimen Status'] == 'Deaccessioned'].index
    d = d.drop(bi)    
    d = d.reset_index()
    
    # Set the sample type
    d['ADORE sample type'] = ''
    for i in range(len(d)):
        spt = d['Specimen Type'].iloc[i]
        if (spt in ['10^7 PBMC', 'Whole Blood', 'Plasma']):
            d['ADORE sample type'].iat[i] = spt.replace(' ', '_')
        else:
            body_site = d['Body Site'].iloc[i]
            if (isinstance(body_site, str)):
                dash_index = body_site.find('-')
                d['ADORE sample type'].iat[i] = \
                    body_site[(dash_index+2):].replace(' ', '_')
                # Add in fat if necessary
                if (('Subcutaneous' in body_site) or
                    ('Omental' in body_site) or
                    ('Visceral' in body_site)):
                    d['ADORE sample type'].iat[i] = body_site[(dash_index+2):] + \
                        '_fat'
 
    # Return    
    return d

def deduce_sample_event(d_redcap, d_oncore, output_folder,
                        match_window_days=10):
    """ Tries to match samples to a visit for each patient """
   
    # Find the unique patients in redcap
    un_redcap_mrns = d_redcap['demo_uk_mrn'].unique()
    
    # Find the event_names in redcap
    all_event_names = d_redcap['redcap_event_name'].unique()
    sample_events = [x for x in all_event_names if ('months' in x)]
    
    print(sample_events)
    
    # Add some columns to d_oncore
    d_oncore['REDCap_patient_found'] = False
    d_oncore['REDCap_visit_day_difference'] = None
    d_oncore['REDCap_visit_type'] = None
   
    for (redcap_i, un_red_mrn) in enumerate(un_redcap_mrns):
        
        # Find the entries in d_oncore for the patient
        vi_oncore = d_oncore.index[d_oncore['Patient ID'] == un_red_mrn].tolist()
        
        # Set the patient_found
        d_oncore.loc[vi_oncore, 'REDCap_patient_found'] = True
        
        for (event_i, sample_ev) in enumerate(sample_events):
            # Find the row for the patient visit
            d_redcap_row = d_redcap[(d_redcap['demo_uk_mrn'] == un_red_mrn) &
                         (d_redcap['redcap_event_name'] == sample_ev)]
            
            # If there is no match, continue
            if (len(d_redcap_row) == 0):
                continue            
            
            # Pull off the date
            redcap_visit_date = d_redcap_row['visit_date'].iloc[0]

            # Set the day diff, and if appropriate assign to event
            for ind in vi_oncore:
                
                d_oncore['REDCap_visit_day_difference'].iat[ind] = \
                    (d_oncore['Collection Date'].iloc[ind] - redcap_visit_date).days

                if (abs(d_oncore['REDCap_visit_day_difference'].iloc[ind]) <=
                        match_window_days):
                    # Assign
                    d_oncore['REDCap_visit_type'].iat[ind] = sample_ev
    
    # Set status for unmatched
    vi_unmatched = d_oncore.index[d_oncore['REDCap_visit_type'].isnull()].tolist()
    d_oncore.loc[vi_unmatched, 'REDCap_visit_type'] = 'Unmatched'
    
    # Drop unnecessary column
    d_oncore = d_oncore.drop(['REDCap_visit_day_difference'], axis=1)
    
    # Create a dataframe with the patients that haven't been found
    d_not_found = d_oncore[d_oncore['REDCap_patient_found'] == False]
    # Save it
    not_found_file_string = os.path.join(output_folder, 'not_found.csv')
    d_not_found.to_csv(not_found_file_string, index=False)
                      
    # Save the main dataframe
    oncore_file_string = os.path.join(output_folder, 'oncore_data.csv')
    d_oncore.to_csv(oncore_file_string, index=False)
    
    return (d_oncore)

def count_patient_samples(d_redcap, d_oncore, output_folder):
    """ Count the samples of each type for each patient """
    
    # Find the unique record_ids
    un_record_ids = d_redcap['record_id'].unique()
    
    # Find the event_names in redcap
    all_event_names = d_redcap['redcap_event_name'].unique()
    sample_events = [x for x in all_event_names if ('months' in x)]
    
    # Add in unmatched
    sample_events.append('Unmatched')
    
    # Get the sample types
    sample_types = d_oncore['ADORE sample type'].unique()
    # Drop the empty one
    sample_types = [x for x in sample_types if not (x=='')]
    
    # Set the col names
    col_names = ['record_id', 'demo_uk_mrn']
    for ty in sample_types:
        for se in sample_events:
            
            # Check for sample types that can only be at 0 months or unmatched
            if ((not (ty in blood_sample_types)) and
                (not (se in ['0_months_arm_1', 'Unmatched']))):
                continue
            
            for st in specimen_statuses:
                # Work out the event string
                under_ind = [i for i, c in enumerate(se) if (c == '_')]
                if (len(under_ind) > 1):
                    se_string = se[0:under_ind[-2]]
                else:
                    se_string = se
                
                col_names.append('%s_%s_%s' % (ty, se_string, st))
       
    # Make a database
    d_counts = pd.DataFrame(columns=col_names, index=range(len(un_record_ids)))
    
    # Loop through patients counting the samples
    for pat_i in range(len(un_record_ids)):
        d_counts['record_id'].iat[pat_i] = un_record_ids[pat_i]
        
        # pull out the patient so we can work out the mrn
        d_pat = d_redcap[d_redcap['record_id'] == un_record_ids[pat_i]]
        # Set the mrn
        uk_mrn = d_pat['demo_uk_mrn'].iloc[0]
        d_counts['demo_uk_mrn'].iat[pat_i] = uk_mrn
        
        # Now cycle through the type / event / status combination
        for ty in sample_types:
            for se in sample_events:
                
                # Check for sample types that can only be at 0 months or unmatched
                if ((not (ty in blood_sample_types)) and
                    (not (se in ['0_months_arm_1', 'Unmatched']))):
                    continue
                
                for st in specimen_statuses:

                    d_match = d_oncore[(d_oncore['Patient ID'] == uk_mrn) &
                                       (d_oncore['ADORE sample type'] == ty) &
                                       (d_oncore['REDCap_visit_type'] == se) &
                                       (d_oncore['Specimen Status'] == st)].copy(deep=True)

                    if (d_match.empty):
                        no_of_samples = 0
                    else:
                        no_of_samples = len(d_match)
                                                
                    # Work out the event string
                    under_ind = [i for i, c in enumerate(se) if (c == '_')]
                    if (len(under_ind) > 1):
                        se_string = se[0:under_ind[-2]]
                    else:
                        se_string = se
                    col_name = '%s_%s_%s' % (ty, se_string, st)
                    
                    # Store it
                    d_counts[col_name].iat[pat_i] = no_of_samples
    
    # Save to folder
    counts_file_string = os.path.join(output_folder, 'sample_counts.csv')
    d_counts.to_csv(counts_file_string, index=False)

    # Convert to import file
    d_import = d_counts.copy(deep=True)

    # Add / delete columns for import
    d_import = d_import.drop('demo_uk_mrn', axis=1)
    d_import['redcap_event_name'] = 'global_arm_1'
    d_import.insert(1, 'redcap_event_name', d_import.pop('redcap_event_name'))

    # Now adjust column names to match REDCap fields
    
    for col in d_import.columns:
        col_string = col
        if not ((col_string == 'record_id') or
                (col_string == 'redcap_event_name')):
            col_string = 'sa_%s' % col_string
        col_string = col_string.replace('_months','mo')
        col_string = col_string.replace('Unmatched','unma')
        col_string = col_string.replace('Available', 'av')
        col_string = col_string.replace('Shipped', 'sh')
        col_string = col_string.replace('10^7_PBMC', 'pbmc')
        col_string = col_string.replace('Whole_Blood', 'whbl')
        col_string = col_string.replace('Plasma', 'plas')
        col_string = col_string.replace('Stomach', 'stom')
        col_string = col_string.replace('Liver', 'live')
        col_string = col_string.replace('Small_Intestine', 'smin')
        col_string = col_string.replace('Subcutaneous_fat', 'subf')
        col_string = col_string.replace('Visceral_fat', 'visf')
        col_string = col_string.replace('Omental_fat', 'omef')
        
        d_import.rename(columns={col: col_string}, inplace=True)

    # Create the import file
    import_file_string = os.path.join(output_folder, 'redcap_import.csv')        
    d_import.to_csv(import_file_string, index=False)
        
def convert_series_to_datetimes(d):
    """ Find columns with the word 'date', convert to Datetime """
    
    # Code
    cols = d.columns
    
    for (i,c) in enumerate(cols):
        if ('date' in c.lower()):
            d[c] = pd.to_datetime(d[c])
    
    return d

def tidy_mrn(mrn_string):
    """ Tidies mrn """
   
    n = len(mrn_string)
    
    while (n < 9):
        mrn_string = '0%s' % mrn_string
        n = len(mrn_string)
        
    return mrn_string

############################################################################
if __name__ == "__main__":
    
    # Parse variables
    redcap_data_file_string = sys.argv[1]
    oncore_report_file_string = sys.argv[2]
    output_folder = sys.argv[3]
    
    # display
    print('REDCap data file: %s' % redcap_data_file_string)
    print('OnCore data file: %s' % oncore_report_file_string)
    print('Output folder: %s' % output_folder)
    
    # Make sure the output folder exists
    if not (os.path.isdir(output_folder)):
        os.makedirs(output_folder)
    
    # Pull the data from REDCap via the API    
    d_redcap = return_REDCap_data(redcap_data_file_string)
    
    print(d_redcap)

    # And load the data from OnCore
    d_oncore = return_OnCore_data(oncore_report_file_string)
    
    print(d_oncore)
    
    # Match the sample collection dates to redcap entries    
    d_oncore = deduce_sample_event(d_redcap, d_oncore, output_folder)
    
    # Count patient samples for each category
    d_redcap = count_patient_samples(d_redcap, d_oncore, output_folder)
    
   
    