import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import altair as alt


#import functions
import pages.page_functions.map_functions as map_func

#set page config
st. set_page_config(layout="wide")

#---------------------------
#load required files
#---------------------------
#LOOK UP FILE PATHS
lookup_imd_decile = 'build_data\\lsoa_lookup\\imd2019lsoa_decile.csv'
loookup_imd_decile_detailed = 'build_data\\lsoa_lookup\\Indices_of_Multiple_Deprivation_(IMD)_2019.csv'
loookup_lsoa_to_utla = 'build_data\\lsoa_lookup\\LSOA_(2011)_to_UTLA_(2021)_Lookup_in_England_and_Wales_.csv'
loookup_lower_tier_to_utla = 'build_data\\lsoa_lookup\\Lower_Tier_LA_to_UTLA__April_2019__Lookup_in_England_and_Wales.csv'
lookup_lsoa_lower_tier_la = 'build_data\\lsoa_lookup\\LSOA_2021_to_Ward_to_Lower_Tier_LA_May_2022.csv'

#POP ESTIMATES FILE PATHS
pop_est_females_syoa = 'build_data\\pop_estimates\\2020_females_pop_lsoa_syoa.csv'
pop_est_males_syoa = 'build_data\\pop_estimates\\2020_males_pop_lsoa_syoa.csv'
pop_est_persons_syoa = 'build_data\\pop_estimates\\2020_persons_pop_lsoa_syoa.csv'

#SHAPEFILES
#lsoa level
shapefile_path_lsoa = 'build_data\\shapefiles\\LSOA_shapefiles\\Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Generalised_Clipped_(BGC)_EW_V3.shp'

#UTLA level
shapefile_path_utla = 'build_data\\shapefiles\\UTLA_shapefiles\\Counties_and_Unitary_Authorities_(December_2021)_EN_BFC.shp'

#POP FORECASTS
pop_proj_path_utla = 'build_data\\pop_projections\\county_unitary_projections_formatted.csv'
pop_proj_path_district = 'build_data\\pop_projections\\district_projections_formatted.csv'

#call function to create dict of necessary reference files
dict_files = map_func.create_data_files(
    lookup_imd_decile,
    loookup_imd_decile_detailed,
    loookup_lsoa_to_utla,
    lookup_lsoa_lower_tier_la,
    pop_est_females_syoa,
    pop_est_males_syoa,
    pop_est_persons_syoa,
    shapefile_path_lsoa,
    shapefile_path_utla,
    pop_proj_path_utla,
    pop_proj_path_district
    )

#create shape file variable from dictionary
gdf_lsoa = dict_files['shapefiles']['gdf_lsoa']

st.title(':blue[Population Visialisation Resource]')
st.subheader('Overview of data sources')
with st.expander(label='Click for overview of data sources'):
    st.write('The data used in this app is all publicy availalbe from NOMIS web (population projections), ONS (population estimates by gender and single year of age), and Open Geography Portal (Shapefiles).')

st.subheader('Set up parameters')


col1, col2, col3 = st.columns(3)
with col1:
    what_map = st.selectbox(
        label='What would you like to visualise?', 
        options=['Deprivation deciles', 'Population density', 'Pop. projection by age'])

#if what_map != 'Pop. projection by age':
with col2:
    geography_level = st.selectbox(
        'Select the level of geography to map', 
        options=['Upper Tier or Unitary Authority', 'District Authority or Place'], 
        index=0)

    if geography_level == 'Upper Tier or Unitary Authority':
        area_text = "Upper Tier or Unitary Authority/ies"
        filter_column = 'UTLA21NM'
        list_options = sorted(list(set(dict_files['lookups']['df_lsoa_to_utla'][filter_column])))
        default_options = ['Derbyshire', 'Derby']

    elif geography_level == 'District Authority or Place':
        area_text = "Place(s) or District Authority/ies"
        filter_column = 'LTLA22NM'
        list_options = sorted(list(set(dict_files['lookups']['df_lsoa_lower_tier'][filter_column])))
        default_options = ['Amber Valley', 'Bolsover', 'Chesterfield', 'Derbyshire Dales', 'Erewash', 'High Peak', 'North East Derbyshire', 'South Derbyshire', 'Derby']

    else:
        st.write('invalid selection')

with col3:
    list_of_areas_to_map = st.multiselect(
        f'Select the {area_text} you want to map', 
        options=list_options,
        default=default_options
        )

if what_map == 'Pop. projection by age':
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        pop_proj_gender = st.selectbox(label='Select the gender for the population forecast', options=['Persons', 'Male', 'Female'])
    with col2:
        pop_proj_age_range = st.selectbox(
            label='Select the age range under consideration', 
            options=[
                'Aged 0 to 15',
                'Aged 16 to 24',
                'Aged 25 to 49',
                'Aged 50 to 64',
                'Aged 65+',
                'All ages'
            ])
    with col3:
        options=[2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]

        pop_proj_baseline_year = st.selectbox(
            label='Select baseline year', 
            options=options,
            index=0)

    with col4:
        remaining_options = map_func.get_remaining_years(options, pop_proj_baseline_year)
        pop_proj_forecast_year = st.selectbox(
            label='Select forecast year', 
            options=remaining_options,
            index=0)

#st.write(dict_files['pop_projections']['pop_proj_utla'].head())
#st.write(dict_files['pop_projections']['pop_proj_district'].head())




#else:
#    with col2:
#        area_text = "Place(s) or District Authority/ies"
#        filter_column = 'LTLA22NM'
#        list_options = sorted(list(set(dict_files['lookups']['df_lsoa_lower_tier'][filter_column])))
#        list_of_areas_to_map = st.multiselect(f'Select the {area_text} you want to map', options=list_options)

col1, col2, col3 = st.columns(3)
#with col1:
    #what_map = st.selectbox(label='What would you like to map?', options=['Deprivation deciles', 'Population density'])

if what_map == 'Population density':
    with col1:
        pop_gender = st.selectbox(label='Select population gender', options=['Male', 'Female', 'Persons'])
    with col2:
        min_age = st.slider(label='Enter min age for reference population', min_value = 1, max_value=99, step=1)
    with col3:
        max_age = st.slider(label='Enter max age for reference population', min_value = 1, max_value=99, step=1)

    df_selected_pop = dict_files['pop_estimates'][pop_gender]

    df_lsoa_summed_age_range = map_func.calculate_age_sum(df_selected_pop, min_age, max_age, 'LSOA Code')


#Build map
#BUILD MAPPING FILE
df_imd_decile = dict_files['lookups']['df_imd_decile']
df_lsoa_to_utla = dict_files['lookups']['df_lsoa_to_utla']
df_lsoa_lower_tier = dict_files['lookups']['df_lsoa_lower_tier']

# Inner join: Include only rows that have matching values in both DataFrames
subset_df_imd_decile = df_imd_decile[['FeatureCode', 'Value']].set_index('FeatureCode')
subset_df_lsoa_to_utla = df_lsoa_to_utla[['LSOA11CD', 'UTLA21CD', 'UTLA21NM']].set_index('LSOA11CD')
subset_df_lsoa_lower_tier = df_lsoa_lower_tier[['LSOA21CD', 'LTLA22CD', 'LTLA22NM']].set_index('LSOA21CD')

joined_df = pd.concat([subset_df_lsoa_to_utla, subset_df_imd_decile], axis=1)

joined_df_final = pd.concat([joined_df, subset_df_lsoa_lower_tier], axis=1)

mask = joined_df_final[filter_column].isin(list_of_areas_to_map)
joined_df_final = joined_df_final[mask]

joined_df_final.reset_index(inplace=True)
joined_df_final.rename(columns={'index': 'LSOACODE'}, inplace=True)

#subset population count data, if this option selected
if what_map == 'Population density':
    joined_df_final = pd.merge(joined_df_final, df_lsoa_summed_age_range, left_on='LSOACODE', right_on='LSOA Code')
    #st.dataframe(df_lsoa_summed_age_range_selected_area)

# Perform the merge
merged_df = pd.merge(gdf_lsoa, joined_df_final, left_on='LSOA11CD', right_on='LSOACODE', how='inner')

# Convert the merged DataFrame back to a GeoDataFrame
filtered_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')

try:
    # Render the plot in Streamlit
    if what_map == 'Deprivation deciles':
        #map_func.render_folium_map(filtered_gdf)
        map_func.render_folium_map_heatmap(filtered_gdf, count_column='Value', line_weight=1, color_scheme='RdYlGn', title='Deprivation Decile Map')
    
    elif what_map == 'Population density':
        title = f'Population density ({pop_gender} aged {min_age}-{max_age})'
        map_func.render_folium_map_heatmap(filtered_gdf, count_column='Age_Sum', line_weight=1, title=title)
        #st.write(filtered_gdf)

    elif what_map == 'Pop. projection by age':
        st.subheader('Population change in selected areas:')

        filtered_df_pop_proj = map_func.filter_dataframe_pop_projections(
            geography_level,
            dict_files, 
            list_of_areas_to_map, 
            pop_proj_baseline_year, 
            pop_proj_forecast_year, 
            pop_proj_gender, 
            pop_proj_age_range
        )

        st.dataframe(filtered_df_pop_proj)
        map_func.visualize_population_change(filtered_df_pop_proj)
except:
    st.stop()
