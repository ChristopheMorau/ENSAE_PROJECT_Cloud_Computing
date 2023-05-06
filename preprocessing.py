import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class Preprocessor:

    def __init__(self, train_path, test_path) -> None:
        """A class to reprocess trainself.df in order to fit the needs of our model
        Args: 
            train_path: the path to find the training trainself.df
            test_path: the path to find the test trainself.df"""
        self.traindf = pd.read_csv(train_path)
        self.traindf = self._duplicates_handler(self.traindf)
        self.testdf = pd.read_csv(test_path)

        self.traindf["is_train"] = True
        self.testdf["is_train"] = False
        self.df = pd.concat([
                self.traindf,
                self.testdf
            ])
        
        del self.traindf
        del self.testdf

    def _duplicates_handler(self, df_dup):
        """Removes lines that describe a same building. Not applied on test set"""
        subset = ['altitude',
          'area_code',
          'building_category',
          'building_height_ft',
          'building_year',
          'building_total_area_sqft',
          'consumption_measurement_date']
        df_dup = df_dup[~df_dup.duplicated(subset=subset, keep='first')]
        return df_dup

    def _nans_handler(self):
        """Takes care of the NaNs. Drops the columns with too many missing values, fill
        the others with likely values"""
        # Too much NA -> drop # Add a missing column value ?
        columns_to_remove = ['nb_gas_meters_commercial', 
                            'nb_gas_meters_housing',
                            'nb_housing_units', 
                            'nb_power_meters_commercial', 
                            'nb_power_meters_housing',
                            'nb_commercial_units']

        self.df = self.df.drop(columns=columns_to_remove, axis='columns')

        # Fill NA with zero
        fill_NA_with_zero = ['nb_parking_spaces',
                            'percentage_glazed_surfaced']
        for column in fill_NA_with_zero:
            self.df[column] = self.df[column].fillna(0)

        fill_NA_with_missing = ['balcony_depth',
                            'outer_wall_thickness']
        for column in fill_NA_with_missing:
            self.df[column] = self.df[column].fillna(0)

        self.df["nb_gas_meters_total"] = self.df["nb_gas_meters_total"].fillna(0)
        self.df.loc[self.df["nb_gas_meters_total"] != 0,"nb_gas_meters_total"] = self.df.loc[self.df["nb_gas_meters_total"] != 0, "nb_gas_meters_total"]

        self.df["nb_power_meters_total"] = self.df["nb_power_meters_total"].fillna(0)
        self.df.loc[self.df["nb_power_meters_total"] != 0,"nb_power_meters_total"] = self.df.loc[self.df["nb_power_meters_total"] != 0, "nb_power_meters_total"]

        self.df["nb_units_total"] = self.df["nb_units_total"].fillna(0)

        # Fill NA with mean
        self.df['nb_meters'] = self.df['nb_meters'].fillna(1.49)

        # Fill NA with median
        self.df['altitude'] = self.df['altitude'].fillna(295.2756)
        self.df['window_heat_retention_factor'] = self.df['window_heat_retention_factor'].fillna(1.13307)
        self.df['window_thermal_conductivity'] = self.df['window_thermal_conductivity'].fillna(16.06)
        self.df['living_area_sqft'] = self.df['living_area_sqft'].fillna(1022)
        # Special treatments
        self.df['has_balcony'] = self.df['has_balcony'].fillna('False')
        self.df['building_year'] = self.df.apply(lambda x: x['building_year'] if not pd.isna(x['building_year'])
                                                        else int((str(x['building_period'])[-4:]
                                                            if str(x['building_period'])[-4:] != " sup"
                                                            else 2000)), axis=1)

        self.df['outer_wall_thickness'] = self.df['outer_wall_thickness'].str[:2]
        self.df['outer_wall_thickness'] = self.df['outer_wall_thickness'].fillna(24.438)
        self.df['outer_wall_thickness'] = self.df['outer_wall_thickness'].astype(int)

        self.df.loc[self.df['upper_floor_material'] == 'concrete slab', 'upper_floor_thermal_conductivity'] = \
            self.df.loc[self.df['upper_floor_material'] == 'concrete slab', 'upper_floor_thermal_conductivity'].fillna(4.401686)
        self.df['upper_floor_thermal_conductivity'] = self.df['upper_floor_thermal_conductivity'].fillna(1.749949)

        self.df.loc[self.df['lower_floor_material'] == 'uninsulated', 'lowe_floor_thermal_conductivity'] = \
            self.df.loc[self.df['lower_floor_material'] == 'uninsulated', 'lowe_floor_thermal_conductivity'].fillna(4.498072)
        self.df['lowe_floor_thermal_conductivity'] = self.df['lowe_floor_thermal_conductivity'].fillna(2.813221)

        self.df.loc[self.df['wall_insulation_type'] == 'non insulated', 'outer_wall_thermal_conductivity'] = \
            self.df.loc[self.df['wall_insulation_type'] == 'non insulated', 'outer_wall_thermal_conductivity'].fillna(10.508490)
        self.df['outer_wall_thermal_conductivity'] = self.df['outer_wall_thermal_conductivity'].fillna(2.406226)


        self.df['building_height_NA'] = self.df['building_height_ft'].isnull().astype(int)
        median_height = 17.060368
        self.df["building_height_ft"] = self.df["building_height_ft"].fillna(median_height)

        self.df['building_total_area_sqft_NA'] = self.df['building_total_area_sqft'].isnull().astype(int)
        median_area = 1205.6
        self.df["building_total_area_sqft"] = self.df["building_total_area_sqft"].fillna(median_area)

    def _columns_reprocessor(self):
        """Handles typing and feature engineering"""
        # New features
        self.df['building_volume'] = self.df['building_height_ft'] * self.df['building_total_area_sqft']
        #building_category
        self.df["n_condo"] = n_condo = self.df.building_category.str.split('condo').str.len()-1
        self.df["n_house"] = self.df.building_category.str.split('individual house').str.len()-1

        #building_class
        self.df["building_class0"] = self.df.building_class.str.split('individual').str.len() - 1
        self.df["building_class1"] = self.df.building_class.str.split('2 to 11').str.len() - 1
        self.df["building_class2"] = self.df.building_class.str.split('12+').str.len() - 1

        #normalize building_year
        # self.df.building_year = 2019 - self.df.building_year

        #turn dates to continuous ints
        max_date = pd.to_datetime(self.df.consumption_measurement_date).max()
        self.df.consumption_measurement_date = (max_date - pd.to_datetime(self.df.consumption_measurement_date)).dt.days

        #heating_energy_source
        keywords = ['charbon', 'electricity', 'gas', 'gpl/butane/propane', 'lpg/butane/propane', 'heatnetwork', 'oil', 'wood']
        self.df["hes_fuel"] = self.df.heating_energy_source.astype(str).apply(lambda x: max(len(x.split('fuel')), len(x.split('fuel-oil'))))
        for idx, keyword in enumerate(keywords):
            self.df[f"hes_class{idx}"] = self.df.heating_energy_source.str.split(keyword).str.len()-1

        #lower_floor_insulation_type
        self.df["is_na_floor_insulation"] = self.df.lower_floor_insulation_type.isna().astype(int) # /!\ may be 'missing' in dataset
        self.df["is_floor_insulated"] = 1 - (self.df.lower_floor_insulation_type.str.find('uninsulated') >= 0).astype(int)
        self.df["is_floor_insulated_external"] = (self.df.lower_floor_insulation_type.str.find('external') >= 0).astype(int)
        self.df["is_floor_insulated_internal"] = (self.df.lower_floor_insulation_type.str.find('internal') >= 0).astype(int)

        #lower_floor_material
        keywords = ['floor between wooden joists with or without infill', 'insulated joist floor', 'concrete slab', 'wooden floor on wooden joists', 'wooden floor on metal joists', 'brick or rubble wall joists', 'non-differing party floor', 'joist on metal joists', 'heavy floor', 'such as clay floor joists', 'concrete beams', 'floor between metal joists with or without infill', 'floor with or without infill', 'shingles and infill']
        for idx, keyword in enumerate(keywords):
            self.df[f"is_lower_floor_material_class{idx}"] = (self.df.lower_floor_material.str.find(keyword) >= 0).astype(int)


    def _encoder(self):   
        """Turns categories into integers values (dummies...)"""

                #main_heat_generators
        self.df = pd.concat([
            self.df,
            pd.get_dummies(self.df.main_heat_generators, drop_first=True)
        ], axis=1) 

        #main_heating_type
        keywords = ['poele ou insert gpl/butane/propane', 'geothermal heat pump', 'indeterminate energy condensing boiler', 'standard lpg/butane/propane boiler', 'air-to-water heat pump', 'water/water heat pump', 'low temperature oil boiler', 'joule effect generators', 'pac indéterminée', 'other indeterminate heating', 'gas radiators', 'standard oil boiler', 'standard gas boiler', 'standard coal boiler', 'wood boiler', 'solar heating', 'indeterminate low temperature energy boiler', 'chaudiere charbon condensation', 'indeterminate stove or insert', 'wood stove or insert', 'air/air heat pump', 'oil stove or insert', 'low temperature gas boiler', 'low temperature lpg/butane/propane boiler', 'null', 'heat network', 'electric boiler', 'chaudiere charbon basse temperature', 'lpg/butane/propane condensing boiler', 'coal stove or insert', 'gas condensing boiler', 'indeterminate wood heating', 'indeterminate energy boiler indeterminate', 'standard indeterminate energy boiler', 'oil condensing boiler']
        for idx, keyword in enumerate(keywords):
            self.df[f"main_heating_type_class{idx}"] = (self.df.main_heating_type.str.find(keyword) >= 0).astype(int)
            
        #main_water_heaters
        tmp = pd.get_dummies(self.df.main_water_heaters, drop_first=True)
        for c in tmp.columns:
            self.df[c] = tmp[c].copy()
        #self.df = self.df.join(pd.get_dummies(self.df.main_water_heaters, drop_first=True))

        #main_water_heating_type
        keywords = ['indeterminate energy condensing boiler', 'standard lpg/butane/propane boiler', 'ecs bois indetermine', 'joule-effect electric water heater', 'low temperature oil boiler', 'independent lpg/butane/propane water heater', 'independent gas water heater', 'standard oil boiler', 'standard gas boiler', 'standard coal boiler', 'low-temperature independent water heater', 'thermodynamic electric hot water (heat pump or storage tank)', 'poele bouilleur bois', 'wood boiler', 'indeterminate indeterminate energy boiler', 'independent water heater indeterminate', 'indeterminate low-temperature energy boiler', 'low temperature gas boiler', 'low temperature lpg/butane/propane boiler', 'heat network', 'indeterminate mixed production', 'other indeterminate dhw', 'electric boiler', 'solar hot water', 'lpg/butane/propane condensing boiler', 'gas condensing boiler', 'chauffe-eau fioul independant']
        for idx, keyword in enumerate(keywords):
            self.df[f"main_water_heating_type_class{idx}"] = (self.df.main_water_heating_type.str.find(keyword)>=0).astype(int)
            
        #outer_wall_materials
        self.df = pd.concat([
            self.df,
            pd.get_dummies(self.df.outer_wall_materials, drop_first=True)
        ], axis=1) 
        
        #radon_risk_level
        self.df.radon_risk_level = self.df.radon_risk_level.map({'low':1, 'medium':2, 'high':3}).fillna(0)

        #roof_material
        keywords = ["tiles", "slate", "zinc aluminum", "concrete"]
        for idx, keyword in enumerate(keywords):
            self.df[f"is_roof_material_class{idx}"] = (self.df.roof_material.str.find(keyword)>=0).astype(int)

        #solar_heating
        self.df = pd.concat([
            self.df,
            pd.get_dummies(self.df.solar_heating, drop_first=True).rename(columns={True: 'is_solar_heating'})
        ], axis=1) 

        #solar_water_heating
        self.df = pd.concat([
            self.df,
            pd.get_dummies(self.df.solar_water_heating, drop_first=True).rename(columns={True: 'is_solar_water_heating'}),
        ], axis=1) 

        #thermal_inertia
        self.df.thermal_inertia = self.df.thermal_inertia.map({'low':1, 'medium':2, 'high':3, 'very high':4}).fillna(0)
        
        #upper_floor_adjacency_type
        self.df.upper_floor_adjacency_type = (self.df.upper_floor_adjacency_type.str.find('LNC') >= 0).astype(int)

        self.df = self.df.drop(columns=[
            "building_category", #separed in two new columns n_condo, n_house
            "building_class", #separed in building_class0, building_class1, building_class2
            "building_use_type_code",
            "has_balcony",
            "heat_generators",
            "heating_energy_source", #separed in 9 binary variables starting with "hes"
            "lower_floor_insulation_type", #separed in 4 binary variables
            "lower_floor_material", #separed in 14 binary variables : is_lower_floor_material_class + idx
            "main_heat_generators", #get-dummied
            "main_heating_type", #separed in 13 binary variables : main_heating_type_class + idx
            "main_water_heaters", #get-dummied (23-1=22 columns)
            "main_water_heating_type", #separed with keywords in 26 binary variables : is_main_water_heating_type_class + idx
            "outer_wall_materials", #get-dummied (20-1= 19 columns)
            "post_code", #area code better
            "renewable_energy_sources", #99% missing
            "roof_material", #separed with keywords in 4 binary variables : is_roof_material_class + idx
            "solar_heating", #binarized
            "solar_water_heating", #binarized
        ])

        categorical = [
            "additional_heat_generators",
            "additional_water_heaters",
            "area_code",
            "balcony_depth",
            "bearing_wall_material",
            "building_period",
            "building_type",
            "building_use_type_description",
            "clay_risk_level",#???
            "has_air_conditioning",
            "heating_type",
            "is_crossing_building",
            "lower_floor_adjacency_type",
            "hes_charbon",
            "hes_electricity",
            "hes_fuel",
            "hes_gas",
            "hes_gpl",
            "hes_lpg",
            "hes_heatnetwork",
            "hes_oil",
            "hes_wood",
            "building_class0",
            "building_class1",
            "building_class2",
        ]

        continuous = [
            "altitude",
            "building_height_ft",
            "building_total_area_sqft",
            "building_year",
            "consumption_measurement_date",
            "living_area_sqft",
            "lowe_floor_thermal_conductivity",
            "nb_commercial_units",
            "nb_dwellings",
            "nb_gas_meters_commercial",
            "nb_gas_meters_housing",
            "nb_gas_meters_total",
            "nb_housing_units",
            "nb_meters",
            "nb_parking_spaces",
            "nb_power_meters_commercial",
            "nb_power_meters_housing",
            "nb_power_meters_total",
            "nb_units_total",
            "outer_wall_thermal_conductivity",
            "percentage_glazed_surfaced",
            "thermal_inertia",
            "radon_risk_level",
            "outer_wall_thickness",
            "n_condo",
            "n_house",
            "window_thermal_conductivity",
            "window_heat_retention_factor",
            "upper_floor_thermal_conductivity",
        ]

        categorical_to_add = [
            'water_heating_type',
            'window_filling_type',
            'window_frame_material',
            'window_glazing_type',
            'upper_floor_material',
            'ventilation_type',
            'additional_heat_generators',
            'additional_water_heaters',
            'balcony_depth',
            'bearing_wall_material',
            'building_period',
            'building_type',
            'building_use_type_description', 
            'heating_type',
            'is_crossing_building',
            'lower_floor_adjacency_type'
        ]

        self.df = pd.concat([self.df]+[
        pd.get_dummies(self.df[categorical_to_one_hot_encode]) for categorical_to_one_hot_encode in categorical_to_add
         ], axis=1) 
        self.df = self.df.drop(columns = categorical_to_add)


    def _binarizer(self):
        mlb = MultiLabelBinarizer()

        strlist_parser = lambda col: pd.Series(
            list([list(set(x)) for x in 
                    self.df[col].astype(str).str.replace('[', '').str.replace(']', '').str.split(',')])
        )
        self.df.window_orientation = self.df.window_orientation.apply(lambda x: x.replace("east or west", "est,west"))
        window_orientation_mlb = pd.DataFrame(mlb.fit_transform(strlist_parser("window_orientation")),columns=mlb.classes_, index=self.df.index)
        self.df = self.df.drop("window_orientation", axis=1)
        self.df = pd.concat([
            self.df,
            window_orientation_mlb
        ], axis=1)

        mlb = MultiLabelBinarizer()

        water_heating_energy_source_mlb = pd.DataFrame(
            mlb.fit_transform(self.df.water_heating_energy_source.str.split(" \+ ").fillna("missing")),
            columns=mlb.classes_,
            index=self.df.index,
        )
        self.df = self.df.drop("water_heating_energy_source", axis=1)
        self.df = pd.concat([
            self.df,
            water_heating_energy_source_mlb
        ], axis=1)

        mlb = MultiLabelBinarizer()

        water_heaters_mlb = pd.DataFrame(
            mlb.fit_transform(self.df.water_heaters.str.split(" \+ ").fillna("missing")),
            columns=mlb.classes_,
            index=self.df.index,
        )
        self.df = self.df.drop("water_heaters", axis=1)

        self.df = pd.concat([
            self.df,
            water_heaters_mlb
        ], axis=1)

        mlb = MultiLabelBinarizer()

        wall_insulation_type_mlb = pd.DataFrame(
            mlb.fit_transform(self.df.wall_insulation_type.str.split("\+").fillna("missing")),
            columns=mlb.classes_,
            index=self.df.index,
        )
        self.df = self.df.drop("wall_insulation_type", axis=1)
        self.df = pd.concat([
            self.df,
            water_heating_energy_source_mlb
        ], axis=1)

        mlb = MultiLabelBinarizer()

        upper_floor_insulation_type_mlb = pd.DataFrame(
            mlb.fit_transform(self.df.upper_floor_insulation_type.str.split("\+").fillna("missing")),
            columns=mlb.classes_,
            index=self.df.index,
        )
        self.df = self.df.drop("upper_floor_insulation_type", axis=1)
        self.df = pd.concat([
            self.df,
            upper_floor_insulation_type_mlb
        ], axis=1)
        
        self.df.clay_risk_level.unique()
        rename_risk_dict = {
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        self.df["clay_risk_level"] = self.df.clay_risk_level.replace(rename_risk_dict).fillna(0)

    def reprocess(self):
        self._nans_handler()
        self._columns_reprocessor()
        self._encoder()
        self._binarizer()

    def save(self, path):
        self.df.to_csv(path, index=False)