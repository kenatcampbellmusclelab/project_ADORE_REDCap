# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:55:22 2024

@author: Campbell
"""

import os
import re

from pathlib import Path

import numpy as np
import pandas as pd

import usaddress


# Variables
old_data_file_string = '../data/old_system/ADOREClinicalData_DATA_LABELS_2024-12-23_1553.csv'
new_fields_file_string = '../data/new_system/redcap_fields.txt'
import_file_string = '../import/import_data.csv'

data_dicts = dict()
data_dicts['demo_sex'] = ['Female',
                          'Male',
                          'Other']
data_dicts['demo_race'] = ['American Indican or Alaskan Native',
                           'Asian',
                           'Black or African American',
                           'Native Hawaiian or Other Pacific Islander',
                           'White']
data_dicts['demo_ethnicity'] = ['Hispanic',
                                'Not Hispanic',
                                'Other']

data_dicts['ae_severity'] = ['Mild',
                             'Moderate',
                             'Severe']

medication_list = ['metformin',
                   'sulfanourea',
                   'statins',
                   'cholesterol meds',
                   'dpp4 inhibitors',
                   'glp1',
                   'glybizide',
                   'invokana',
                   'lisinopril']


def create_import_from_orig_data(old_data_file_string,
                                 new_fields_file_string,
                                 import_file_string):
    
    # Get directory of this file
    this_dir = Path(__file__).parent.absolute()
    
    # Correct old_data_file_string for the path
    old_data_file_string = os.path.join(this_dir, old_data_file_string)
    
    # Load the data    
    old_data = pd.read_csv(old_data_file_string,
                           converters={'UK MRN ': str})
    
    # Correct badly formatted fields
    for c in old_data.columns:
        old_data = old_data.rename(columns={c: c.rstrip()})
    
    # Fill empty entries with NaN to then forward fill the MRN
    old_data['UK MRN'] = old_data['UK MRN'].replace('', np.nan)
    old_data['UK MRN'] = old_data['UK MRN'].ffill()

    # Find the unique patients
    old_unique_mrns = old_data['UK MRN'].unique()
    
    # Load the new fields and make an empty dataframe
    with open(new_fields_file_string, 'r') as f:
        temp = f.readlines()
        new_fields = [x[0:-1] for x in temp ]
        
    new_data = pd.DataFrame(columns = new_fields)
    
    # Cycle through the unique patients
    for (pat_index, old_un_mrn) in enumerate(old_unique_mrns):
        
        # Pull out the patient data
        d_pat = old_data[(old_data['UK MRN'] == old_un_mrn)].copy(deep=True)

        # Set patient_id
        pat_id = pat_index + 1
                         
        # Extract the baseline and use that to generate overarching fields
        d_baseline = d_pat[(d_pat['Event Name'] == 'Baseline')].copy(deep=True)
        
        new_data = set_demo_data(d_baseline, pat_id, new_data)
        new_data = set_contact_data(d_baseline, pat_id, new_data)
        
        # Find the unique events for the patient
        pat_unique_events = d_pat['Event Name'].unique()
        
        # Cycle through them
        for visit_index in range(len(pat_unique_events)):
            
            # Find all the patients rows that match the event
            d_pat_event = \
                d_pat[(d_pat['Event Name'] == pat_unique_events[visit_index])]

            # Compress them into one row
            d_pat_event = d_pat_event.infer_objects(copy=False).bfill()
            d_visit = d_pat_event.iloc[:1]
            
            d_visit.to_csv('d:/temp/new_data_2.csv', sep=',')
                        
            # Deduce the event
            old_event_name = d_visit['Event Name'].iloc[0]
            
            if (old_event_name == 'Baseline'):
                new_event_id = '0_months_arm_1'
            elif (old_event_name == '3 Month'):
                new_event_id = '3_months_arm_1'
            elif (old_event_name == '6 Month'):
                new_event_id = '6_months_arm_1'
            elif (old_event_name == 'Year 1'):
                new_event_id = '12_months_arm_1'
                
            # Set data
            new_data = set_visit_data(d_visit, pat_id, new_data, new_event_id)
            new_data = set_clin_data(d_visit, new_data)
            new_data = set_med_hist_data(d_visit, new_data)
            
            # Check for adverse event
            new_data = set_adverse_event_data(d_visit, pat_id, new_data)
        
        # if (pat_index == 0):
        #     break
        
        
    new_data.to_csv('d:/temp/new_data.csv', sep=',', index=False)
    
def set_adverse_event_data(d_visit, pat_id, d_import):
    """ Checks for adverse event and a new row to the import data
        if required """
    
    # Check date, which is empty (float) if not there
    if not (isinstance(d_visit['Start Date'].iloc[0], float)):
           
        # Add a row
        d_import.loc[len(d_import), 'record_id'] = pat_id
        
        # Get the row
        event_row = len(d_import) - 1
        
        # Set the type and instance
        d_import['redcap_event_name'].iat[event_row] = \
            'global_arm_1'
        d_import['redcap_repeat_instrument'].iat[event_row] = \
            'adverse_events'
        d_import['redcap_repeat_instance'].iat[event_row] = 'new'
        
        # And the data
        d_import['ae_comments'].iat[event_row] = strip_commas(
            d_visit['Adverse Effects?']).iloc[0]
        d_import['ae_start_date'].iat[event_row] = \
            d_visit['Start Date'].iloc[0]
        d_import['ae_end_date'].iat[event_row] = \
            d_visit['End Date'].iloc[0]
        d_import['ae_severity'].iat[event_row] = \
            return_unit_off_index_for_key(
                data_dicts['ae_severity'], d_visit['Severity?'].iloc[0])
        d_import['ae_outcome'].iat[event_row] = strip_commas(
            d_visit['Outcome?'].iloc[0])
        
    return d_import
    
        
def set_visit_data(d_visit, pat_id, d_import, event_id):
    """ Adds a new row for the visit """
    
    # Add a row for the patient
    d_import.loc[len(d_import), 'record_id'] = pat_id
    
    # Get the row
    visit_row = len(d_import) - 1
    
    # Set the event name
    d_import['redcap_event_name'].iat[visit_row] = event_id
    d_import['visit_date'].iat[visit_row] = d_visit["Today's Date"].iloc[0]
    
    if (isinstance(d_visit['Date of Liver Biopsy'].iloc[0], float)):
        d_import['visit_liver_procured'].iat[visit_row] = 0
    else:
        d_import['visit_liver_procured'].iat[visit_row] = 1
        
    if (isinstance(d_visit['Date of Fat biopsy'].iloc[0], float)):
        d_import['visit_fat_procured'].iat[visit_row] = 0
    else:
        d_import['visit_fat_procured'].iat[visit_row] = 1
        d_import['visit_fat_mass_g'].iat[visit_row] = d_visit['Grams of Fat'].iloc[0]
        
    if (isinstance(d_visit['Date of blood draw'].iloc[0], float)):
        d_import['visit_blood_procured'].iat[visit_row] = 0
    else:
        d_import['visit_blood_procured'].iat[visit_row] = 1
        d_import['visit_blood_vol_ml'].iat[visit_row] = \
            d_visit['Volume of blood draw (ml)'].iloc[0]
    
    return (d_import)

def set_clin_data(d_visit, d_import):
    
    # Set the row
    visit_row = len(d_import) - 1
    
    # Set the data
    d_import['clin_smoked_before_visit'].iat[visit_row] = \
        return_yes_no_value(
            d_visit['Did the subject smoke in the last 12 hours?'].iloc[0])
    d_import['clin_nsaid_before_visit'].iat[visit_row] = \
        return_yes_no_value(
            d_visit['Did the subject take Aspirin/NSAIDs in the last 72 hours?'].iloc[0])
    d_import['clin_hemat'].iat[visit_row] = \
        d_visit['Hematocrit Levels'].iloc[0]
    d_import['clin_tsh'].iat[visit_row] = \
        d_visit['TSH'].iloc[0]
    d_import['clin_ast'].iat[visit_row] = \
        d_visit['AST'].iloc[0]
    d_import['clin_alt'].iat[visit_row] = \
        d_visit['ALT'].iloc[0]
    d_import['clin_hba1c'].iat[visit_row] = \
        d_visit['HbA1c'].iloc[0]
    d_import['clin_total_chol'].iat[visit_row] = \
        d_visit['Total cholesterol'].iloc[0]
    d_import['clin_hdl'].iat[visit_row] = \
        d_visit['HDL'].iloc[0]
    d_import['clin_ldl'].iat[visit_row] = \
        d_visit['LDL'].iloc[0]
    d_import['clin_trig'].iat[visit_row] = \
        d_visit['Triglycerides'].iloc[0]
    d_import['clin_bp_syst'].iat[visit_row] = \
        d_visit['Blood Pressure (Systolic)'].iloc[0]
    d_import['clin_bp_diast'].iat[visit_row] = \
        d_visit['Blood Pressure (Diastolic)'].iloc[0]
    d_import['clin_pulse_rate'].iat[visit_row] = \
        d_visit['Pulse'].iloc[0]
    d_import['clin_resp_rate'].iat[visit_row] = \
        d_visit['Respiratory Rate'].iloc[0]
    d_import['clin_height_cm'].iat[visit_row] = \
        d_visit['Height (In total cm)'].iloc[0]
    d_import['clin_weight_kg'].iat[visit_row] = \
        d_visit['Weight (in kg)'].iloc[0]
    d_import['clin_bmi'].iat[visit_row] = \
        d_visit['BMI'].iloc[0]
        
    return (d_import)

def set_med_hist_data(d_visit, d_import):
    
    # Set the row
    visit_row = len(d_import) - 1
    
    # Set the data
    d_import['mh_other_study'].iat[visit_row] = \
        return_yes_no_value(
        d_visit['Is Patient enrolled in another study? (i.e. PI, BP, PDT2D, INDIGO, and Tirzepitide)'].iloc[0])
    d_import['mh_other_study_names'].iat[visit_row] = strip_commas(
        d_visit['Study Name'].iloc[0])
    d_import['mh_other_study_ids'].iat[visit_row] = strip_commas(
        d_visit['Study ID'].iloc[0])    
    
    d_import['mh_tobacco_current'].iat[visit_row] = \
        return_yes_no_value(
        d_visit['Do you smoke?'].iloc[0])
    d_import['mh_alcohol_current'].iat[visit_row] = \
        return_yes_no_value(
        d_visit['Do you consume alcoholic beverages?'].iloc[0])
    d_import['mh_alcohol_current_comments'].iat[visit_row] = \
        strip_commas(
            d_visit['How often do consume alcoholic beverages?'].iloc[0])
    d_import['mh_hep_hiv'].iat[visit_row] = \
        return_yes_no_value(
            d_visit['Do you have or have you ever had Hep B, Hep C, HIV or AIDs?'].iloc[0])
    d_import['mh_autoimmune_disease'].iat[visit_row] = \
        return_yes_no_value(
            d_visit["Do you have or have you ever been diagnosed with an autoimmune or inflammatory disease (ex. Type I Diabetes, Crohn's, IBS, rheumatoid arthritis, psoriasis, asthma, lupus, celiac disease, Sjogren's, multiple sclerosis, alopecia, vitiligo, Graves')?"].iloc[0])        
    
    # And now the med comments
    d_import['mh_medical_history_comments'].iat[visit_row] = strip_commas(
        d_visit['Medical History? (diabetes, prediabetes, high blood pressure, kidney disease, heart attack, other)'].iloc[0])

    # Do the medication checkboxes as a loop
    drug_fields = [col for col in d_visit.columns if 
                   (col.startswith('Medication list') & (not 'Other' in col))]
    
    for (i, df) in enumerate(drug_fields):
        eq_index = df.find('=')
        drug_name = df[eq_index+1:-1].lower()
        med_index = medication_list.index(drug_name)
        if (d_visit[df].iloc[0] == 'Checked'):
            import_field = 'mh_medication_checkboxes___%i' % (med_index+1)
            d_import[import_field].iat[visit_row] = 1

    # Back to easier things    
    d_import['mh_medications_other'].iat[visit_row] = \
        strip_commas(
            d_visit['Are you on medications (including steroids, ibuprofen/anti-inflammatories)?'].iloc[0])
    
    d_import['mh_allergies'].iat[visit_row] = \
        strip_commas(
            d_visit['Any allergies? (latex, lidocaine)'].iloc[0])
        
    d_import['mh_surgical_history'].iat[visit_row] = \
        strip_commas(
            d_visit['Surgical history (last 10 years)?'].iloc[0])
        
    # Exercise is a bit awkward
    ex_field = d_visit['Do you exercise? How often (type, duration)'].iloc[0]
    if (isinstance(ex_field, str)):
        d_import['mh_exercise'].iat[visit_row] = 1
        d_import['mh_exercise_comments'].iat[visit_row] = \
            strip_commas(ex_field.lower().title())
            
    # Cold-like is too
    cold_field = d_visit['Have you had a cold/flu/ COVID in the last two weeks? If yes, when?'].iloc[0]
    if (isinstance(cold_field, str)):
        if (('denies' in cold_field.lower()) or
            ('no' in cold_field.lower())):
            d_import['mh_cold_like'].iat[visit_row] = 0
        else:
            d_import['mh_cold_like'].iat[visit_row] = 1
            d_import['mh_cold_like_comments'].iat[visit_row] = \
                strip_commas(cold_field.lower().title())
    
    return d_import   
    
    
    
def set_demo_data(d_patient, pat_id, d_import):
    """ Set demographics for the patient """
    
    # Add a row for the patient
    d_import.loc[len(d_import), 'record_id'] = pat_id
    
    # Get the patient row
    pat_row =len(d_import) - 1
    
    # Set the other fields
    d_import['redcap_event_name'].iat[pat_row] = 'global_arm_1'
    d_import['demo_uk_mrn'].iat[pat_row] = d_patient['UK MRN'].iloc[0]
    d_import['demo_enrollment_date'].iat[pat_row] = d_patient["Today's Date"].iloc[0]
    
    # Name
    name_bits = d_patient["Participant's Name"].iloc[0].split(' ')
    d_import['demo_given_name'].iat[pat_row] = name_bits[0].title()
    # Special case
    if (name_bits[-1].startswith('Jr')):
        d_import['demo_family_name'].iat[pat_row] = name_bits[-2].title() + \
            ' Jr'
    else:
        d_import['demo_family_name'].iat[pat_row] = name_bits[-1].title()
        if (len(name_bits) > 2):
            d_import['demo_initials'].iat[pat_row] = name_bits[1][0]
        
    # Others
    d_import['demo_date_of_birth'].iat[pat_row] = d_patient['Date of Birth'].iloc[0]
    d_import['demo_sex'].iat[pat_row] = return_unit_off_index_for_key(
        data_dicts['demo_sex'],
        d_patient['Gender'].iloc[0])
    
    # Special case for race
    pat_race = d_patient['Race'].iloc[0]
    if pat_race.startswith('Caucasian'):
        pat_race = 'White'
    elif pat_race.startswith('African'):
        pat_race = 'Black or African American'
    uo_index = return_unit_off_index_for_key(
        data_dicts['demo_race'],
        pat_race)
    temp_string = 'demo_race___%i' % uo_index
    d_import[temp_string].iat[pat_row] = '1'
    
    # Hispanic
    hisp_string = d_patient['Ethnicity'].iloc[0]
    if (hisp_string == 'Non-Hispanic'):
        hisp_string = 'Not Hispanic'
    d_import['demo_ethnicity'].iat[pat_row] = return_unit_off_index_for_key(
        data_dicts['demo_ethnicity'],
        hisp_string)
    
    # Planning to be at UK
    plan_to_stay = d_patient['Are you planning on being in the UK area for the next 3 years?'].iloc[0]
    if (plan_to_stay == 'Yes'):
        d_import['demo_plan_at_uk'].iat[pat_row] = 1
    else:
        d_import['demo_plan_at_uk'].iat[pat_row] = 0
    
    return (d_import)

def set_contact_data(d_patient, pat_id, d_import):
    """ Sets the contact data for the patient """
    
    # Find the row for the patient
    pat_row = d_import.index[(d_import['record_id'] == pat_id) & \
            (d_import['redcap_event_name'] == 'global_arm_1')].to_list()[0]

    d_import['contact_phone_number'].iat[pat_row] = \
        d_patient['Phone Number'].iloc[0]
    d_import['contact_email'].iat[pat_row] = \
            d_patient['Email Address'].iloc[0].lower()
       
    # Parse the address
    add_tuple = usaddress.tag(d_patient['Address'].iloc[0])
    add = add_tuple[0]
    add_type = add_tuple[1]
    
    # Special case
    try:
        if not ('PlaceName' in add):
            split_StreetName = add['StreetName'].split(' ')
            if (len(split_StreetName) == 2):
                add['StreetName'] = split_StreetName[0]
                add['PlaceName'] = split_StreetName[1]
                add['StateName'] = add['StreetNamePostType']
            else:
                add['PlaceName'] = ''
                add['StateName'] = ''
        if not ('ZipCode' in add):
            add['ZipCode'] = ''

    except:
        []            
    
    if (add_type == 'Street Address'):
        d_import['contact_street_address_1'].iat[pat_row] = \
            '%s %s %s' % (add['AddressNumber'], add['StreetName'].lower().title(),
                          add['StreetNamePostType'].lower().title())
    elif (add_type == 'PO Box'):
        d_import['contact_street_address_1'].iat[pat_row] = \
            '%s %s' % (add['USPSBoxType'], add['USPSBoxID'])
    d_import['contact_city'].iat[pat_row] = add['PlaceName'].lower().title()
    d_import['contact_state'].iat[pat_row] = add['StateName']
    d_import['contact_zip'].iat[pat_row] = add['ZipCode']
    
    # Parse the emergency contact        
    em_contact = str(d_patient['Emergency Contact Name'].iloc[0])
    
    try:
        relationship = re.search(r'\((.*?)\)',em_contact)
        if (relationship):
            d_import['contact_emergency_name'].iat[pat_row] = \
                em_contact[0 : relationship.start() - 1].lower().title()
            d_import['contact_emergency_relationship'].iat[pat_row] = \
                em_contact[relationship.start()+1 : relationship.end()-1].lower().title()
    except:
        d_import['contact_emergency_name'].iat[pat_row] = \
            str(d_patient['Emergency Contact Name'].iloc[0]).lower().title()
        d_import['contact_emergency_relationship'].iat[pat_row] = ''
    
    d_import['contact_emergency_phone'].iat[pat_row] = \
        d_patient['Emergency Contact phone number'].iloc[0]
        
    # Return
    return (d_import)

def return_unit_off_index_for_key(dictionary, keyword):
    
    uo_index = dictionary.index(keyword) + 1
    
    return uo_index

def return_yes_no_value(keyword):
    """ Returns 1 for yes and 0 for no """
    
    return_value = ''
    
    if (isinstance(keyword, str)):
        if (keyword.lower() == 'yes'):
            return_value = 1
        else:
            return_value = 0
            
    return return_value
    
def strip_commas(var):
    """ Strips commas from a variable if it is a string """
    
    if (isinstance(var, str)):
        var = var.replace(',', ' ')
        
    return var

if __name__ == "__main__":
    create_import_from_orig_data(old_data_file_string,
                                 new_fields_file_string,
                                 import_file_string)
    


