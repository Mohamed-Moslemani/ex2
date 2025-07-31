import pandas as pd
from datetime import timedelta
from holidays import CountryHoliday

class HolidayEnricher:
    def __init__(self, df: pd.DataFrame, date_col: str = "week", country_code: str = "US"):
        self.df = df.copy()
        self.date_col = date_col
        self.country_code = country_code

    def add_public_holidays(self) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        years = self.df[self.date_col].dt.year.unique()
        hdays = CountryHoliday(self.country_code, years=years)

        is_holiday_week = []
        for start_date in self.df[self.date_col]:
            week_dates = [start_date + timedelta(days=i) for i in range(7)]
            has_holiday = any(day in hdays for day in week_dates)
            is_holiday_week.append(has_holiday)

        self.df["is_public_holiday"] = is_holiday_week
        return self.df
