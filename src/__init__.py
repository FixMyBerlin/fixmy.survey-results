import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

this.list_columns = ["bikeReasons", "vehiclesOwned",
                     "whyBiking", "introSelection"]
this.dict_columns = ["berlinTraffic",
                     "motivationalFactors",
                     "transportRatings",
                     "responsible",
                     "climateTraffic",
                     "sharingConditions",
                     "sharingModes",
                     "saveSpace",
                     "annoyingPeople"]
