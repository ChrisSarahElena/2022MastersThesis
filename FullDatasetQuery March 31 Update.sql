select * into #temporarymaster FROM [master].[dbo].[automatic_weather_stations_inmet_brazil_2000_2021]

UPDATE #Temporarymaster
SET [PRECIPITACAO TOTAL HORARIO (mm)] = NULL
where [PRECIPITACAO TOTAL HORARIO (mm)] = '-9999.0' or [PRECIPITACAO TOTAL HORARIO (mm)] = ''

UPDATE #Temporarymaster
SET [TEMPERATURA DO AR - BULBO SECO, HORARIA (C)] = NULL
where [TEMPERATURA DO AR - BULBO SECO, HORARIA (C)]= '-9999.0' OR [TEMPERATURA DO AR - BULBO SECO, HORARIA (C)] = ''

UPDATE #Temporarymaster
SET [TEMPERATURA MAXIMA NA HORA ANT  (AUT) (C)] = NULL
where [TEMPERATURA MAXIMA NA HORA ANT  (AUT) (C)] = '-9999.0' OR [TEMPERATURA MAXIMA NA HORA ANT  (AUT) (C)] = ''

UPDATE #Temporarymaster
SET [TEMPERATURA MINIMA NA HORA ANT  (AUT) (C)] = NULL
where [TEMPERATURA MINIMA NA HORA ANT  (AUT) (C)] = '-9999.0' OR [TEMPERATURA MINIMA NA HORA ANT  (AUT) (C)] = ''

--Pull from the daily weather data and aggregate it by day into sums, avg, min, max, etc.
select distinct [data (YYYY-MM-DD)] as 'Date', Estacao,sum(convert(decimal (20,15),[precipitacao total horario (mm)])) as 'Sum of Precipitation',
max(convert(decimal (20,15),[precipitacao total horario (mm)])) as 'Maximum Precipitation',
min(convert(decimal(20,15),[TEMPERATURA DO AR - BULBO SECO, HORARIA (C)])) as 'Min Temperature Bulbo Seco',
max(convert(decimal(20,15),[TEMPERATURA DO AR - BULBO SECO, HORARIA (C)])) as 'Max Temperature Bulbo Seco',
min(convert(decimal(20,15),[TEMPERATURA MINIMA NA HORA ANT  (AUT) (C)])) as 'Min Temperature Regular',
max(convert(decimal(20,15),[TEMPERATURA MAXIMA NA HORA ANT  (AUT) (C)])) as 'Max Temperature Regular',
avg(convert(decimal(20,15),[TEMPERATURA DO AR - BULBO SECO, HORARIA (C)])) as 'Average Daily Bulb Seco Temperature - For STDev Calc'
INTO #DAILYSET
FROM #temporarymaster
group by [Data (YYYY-MM-DD)], [Estacao]
order by [data (YYYY-MM-DD)]

--Aggregating monthly data based on overall set (just to get the monthly avg temperature)
select 
case when month([Data (YYYY-MM-DD)]) < 10 then concat(year([Data (YYYY-MM-DD)]) ,'.0',month([Data (YYYY-MM-DD)]) )
else concat(year([Data (YYYY-MM-DD)]),'.',month([Data (YYYY-MM-DD)])) end as 'Year.Month',
ESTACAO,
SUM(CONVERT(DECIMAL(20,15),[TEMPERATURA DO AR - BULBO SECO, HORARIA (C)]))/COUNT([TEMPERATURA DO AR - BULBO SECO, HORARIA (C)]) AS 'Average Temperature Bulbo Seco'
into #temps
from #temporarymaster
GROUP BY MONTH([DATA (YYYY-MM-DD)]),YEAR([DATA (YYYY-MM-DD)]),ESTACAO


--The first step of the query is to aggregate all the weather data by month. 
select *, month([date]) as 'Month', year([Date]) as 'Year', [Last Day of Month] = 
eomonth([Date]) into #addmonth from #dailyset

select [Year],[Month], [Estacao],
case when [month] < 10 then concat([Year],'.0',[Month]) else concat([Year],'.',[Month]) end as 'Year.Month',
[Daily Average Precipitation] = sum([Sum of Precipitation])/day([last day of month]),
[Total Monthly Precipitation] = sum([Sum of Precipitation]),
[Maximum Hourly Precipitation] = max([maximum precipitation]),
[Maximum Daily Precipitation] = max([sum of precipitation]),
[Standard Deviation Precipitation] = stdev([Sum of Precipitation]),
[Maximum Hourly Regular Temperature] = max([Max Temperature Regular]),
[Minimum Hourly Regular Temperature] = min([Min Temperature Regular]),
[Maximum Hourly Bulbo Seco] = max([Max Temperature Bulbo Seco]),
[Minimum Hourly Bulbo Seco] = min([Min Temperature Bulbo Seco]),
stdev([Average Daily Bulb Seco Temperature - For STDev Calc]) as 'Standard Deviation Bulbo Seco Daily Temperature'
into #monthaggregate
from #addmonth
group by [Year],[Month], [Estacao],[Last day of month]

select a.[Year.Month],a.[Estacao],
a.[Daily Average Precipitation],
a.[Total Monthly Precipitation],
a.[Maximum Hourly Precipitation],
a.[Maximum Daily Precipitation],
a.[Standard Deviation Precipitation],
a.[Maximum Hourly Regular Temperature],
a.[Minimum Hourly Regular Temperature], 
a.[Maximum Hourly Bulbo Seco],
a.[Minimum Hourly Bulbo Seco],
a.[Standard Deviation Bulbo Seco Daily Temperature],
b.[average temperature bulbo seco]
into #monthaggregate2
From #monthaggregate a
left join #temps b on a.[Year.Month] = b.[Year.Month] and a.estacao = b.estacao 

select * into #updatedmonthaggregate from #monthaggregate2
where [daily average precipitation] is not null;


--Now we can start thinking about how to join them and calculate distances.

	--Months Table
	WITH nums AS
   (SELECT 1 AS value
    UNION ALL
    SELECT value + 1 AS value
    FROM nums
    WHERE nums.value <= 11)
SELECT value as 'Month' INTO #MONTHS
FROM nums;    

	--Years Table
	WITH nums AS
   (SELECT 2000 AS value
    UNION ALL
    SELECT value + 1 AS value
    FROM nums
    WHERE nums.value <= 2019)
SELECT value as 'Years' INTO #Years
FROM nums;   

--Codigo
select distinct[codigo] into #codigo from users.sabell_weatherupload

--Crossjoin
select * into #monthyearcodigo
from #months, #years, #codigo

select *,[One Month Ago] = case when month_int = 1 then concat([YEAR]-1,'.',12) 
when month_int <= 10 then concat([year],'.0',[month_int]-1)
else concat([year],'.',[month_int]-1) end,
[Two Months Ago] = 
case when month_int = 2 then concat([year]-1,'.',12) 
when month_int = 1 then concat([year]-1,'.',11)
when month_int <= 11 then concat([year],'.0',[month_int]-2)
else concat([year],'.',[month_int]-2) end,
[Three Months Ago] = case when month_int = 3 then concat([year]-1,'.',12) 
when month_int = 2 then concat([year]-1,'.',11)
when month_int = 1 then concat([year]-1,'.',10)
when month_int <= 12 then concat([year],'.0',[month_int]-3)
else concat([year],'.',[month_int]-3) end
into #addpasttwomonths
from users.sabell_datadisasternew

select [YearMonth], [Municipality], [Latitude], [Longitude]
into #yearsandmonthsweneed
from #addpasttwomonths
UNION Select [One Month Ago], [Municipality], [Latitude], [Longitude]
from #addpasttwomonths
UNION SELECT [Two Months Ago], [Municipality], [Latitude], [Longitude]
from #addpasttwomonths

--Now we've added the geolocation to the data we have
select a.*,b.[Regiao],b.[UF],b.[Codigo],b.[Latitude],b.[Longitude],b.[Altitude]
into #aggregateddatawithgeolocation
From #updatedmonthaggregate a
left join users.sabell_Weatherupload b on a.[Estacao] = b.codigo


select [Year.Month], [Estacao], [Regiao], [UF], [Latitude], [Longitude], [Altitude] 
into #aggregateddataforjoining
from #aggregateddatawithgeolocation

--We have a set that has all the location, month and year combos we need to have
--We have a set that has all the location, month and year combos we DO have
--We need to find the combinations of those pairs that make the most sense.

--Join all the latitude/longitude pairs from the station set to all the latitude/longitude pairs in the disaster set
select a.latitude as 'Station Latitude',a.longitude as 'Station Longitude',a.Estacao as 'Station Municipality', a.[Codigo] as 'Station Code',
b.latitude as 'Disaster Latitude', b.longitude as 'Disaster Longitude', b.Municipality as 'Disaster Municipality', [YearMonth]
into users.sabell_crossjoin
from users.sabell_weatherupload a
cross join #yearsandmonthsweneed b



--So now we take everything from the crossjoin, and we match to the aggregated data. 
--If there's no data for that particular latitude and longitude in that particular month, then we remove that line from the set.

select * 
into #filteredcrossjoin
From users.sabell_crossjoin a
left join #aggregateddatawithgeolocation b on [Station Latitude] = [latitude] and [station longitude] = [longitude] 
and [Station Code] = b.[estacao]
and [yearmonth] = [year.month]
where b.[estacao] is not null and [disaster latitude] <> ''


--DBCC SHRINKDATABASE (master, 10);  
--GO  

--Calculate the distances between stations and disaster points
SELECT [Station Latitude], [Station Longitude], [Station Municipality], [Disaster Latitude], [Disaster Longitude], [Disaster Municipality],[Year.Month],
Geography::Point([Station Latitude], [Station Longitude], 4326).STDistance(Geography::Point([Disaster Latitude], [Disaster Longitude], 4326)) as 'Distance'
into #temp
from #filteredcrossjoin

--Take the pairs with the minimum distance between them.
select [Disaster Latitude], [Disaster Longitude], [Disaster Municipality], [Year.Month],min(Distance) as 'Min Distance'
into #mindistances
from #temp
group by [Disaster Latitude], [Disaster Longitude], [Disaster Municipality],[Year.Month]

select a.[Disaster Latitude], a.[Disaster Longitude], a.[Disaster Municipality], [Min Distance], b.[Station Latitude], b.[Station Longitude], b.[Station Municipality],a.[Year.Month]
into #matchedset
from #mindistances a
left join #temp b on [Min Distance] = [Distance] and a.[Year.Month] = b.[Year.Month]

select a.*,b.[codigo] into #matchedsetwithcodigo
from #matchedset a
left join users.sabell_weatherupload b on a.[station municipality] = b.[estacao]

select a.*, b.[Disaster Latitude], b.[Disaster Longitude], b.[Disaster Municipality],
b.[Min Distance], b.[Station Latitude], b.[Station Longitude], b.[Station Municipality]
into #weatherset
from #aggregateddatawithgeolocation a
left join #matchedsetwithcodigo b on a.[estacao] = b.[codigo] and a.[year.month] = b.[year.month]


select *, 
[Prior Month] = case when a.month_int = 1 then concat(a.[YEAR]-1,'.',12) 
when a.month_int <= 10 then concat(a.[year],'.0',a.[month_int]-1)
else concat(a.[year],'.',a.[month_int]-1) end,
[Two Months Prior] = case when month_int = 2 then concat([year]-1,'.',12) 
when month_int = 1 then concat([year]-1,'.',11)
when month_int <= 11 then concat([year],'.0',[month_int]-2)
else concat([year],'.',[month_int]-2) end
into #updatedatadisasternew 
from users.sabell_datadisasternew a

--Starting with the disaster dataset, join to the precipitation data.
select a.*,
b.[Daily Average Precipitation] as 'One Month Prior: Daily Average Precipitation',
b.[Total Monthly Precipitation] as 'One Month Prior: Total Monthly Precipitation',
b.[Maximum Hourly Precipitation] as 'One Month Prior: Maximum Hourly Precipitation',
b.[Maximum Daily Precipitation] as 'One Month Prior: Maximum Daily Precipitation',
b.[Maximum Hourly Regular Temperature] as 'One Month Prior: Maximum Hourly Regular Temperature',
b.[Minimum Hourly Regular Temperature] as 'One Month Prior: Minimum Hourly Regular Temperature',
b.[Maximum Hourly Bulbo Seco] as 'One Month Prior: Maximum Hourly Bulbo Seco',
b.[Minimum Hourly Bulbo Seco] as 'One Month Prior: Minimum Hourly Bulbo Seco',
b.[Standard Deviation Precipitation] as 'One Month Prior: Standard Deviation Precipitation',
b.[Standard Deviation Bulbo Seco Daily Temperature] as 'One Month Prior: Standard Deviation Bulbo Seco Daily Temperature',
b.[average temperature bulbo seco] as 'One Month Prior: Average temperature bulbo seco',
b.[Altitude] as 'One Month Prior: Altitude',
b.[Regiao] as 'One Month Prior: Regiao',
b.[UF] as 'One Month Prior: UF',
b.[Station Latitude] as 'One Month Prior: Station Latitude',
b.[Station Longitude] as 'One Month Prior: Station Longitude',
b.[Station Municipality] as 'One Month Prior: Station Municipality',
b.[Min Distance] as 'One Month Prior: Min Distance (m)',
b.[Min Distance]/1000.0 as 'One Month Prior: Min Distance (km)'
into #onemonthcombination from #updatedatadisasternew a 
left join #weatherset b on a.[Prior Month] = b.[Year.Month] and
convert(decimal(20,15),a.[latitude]) = b.[disaster latitude] and convert(decimal(20,15),a.[longitude]) = b.[disaster longitude] and a.[municipality] = b.[disaster Municipality]
where a.[latitude] is not null and a.[latitude] <> ''

select a.*,
b.[Daily Average Precipitation] as 'Two Months Prior: Daily Average Precipitation',
b.[Total Monthly Precipitation] as 'Two Months Prior: Total Monthly Precipitation',
b.[Maximum Hourly Precipitation] as 'Two Months Prior: Maximum Hourly Precipitation',
b.[Maximum Daily Precipitation] as 'Two Months Prior: Maximum Daily Precipitation',
b.[Maximum Hourly Regular Temperature] as 'Two Months Prior: Maximum Hourly Regular Temperature',
b.[Minimum Hourly Regular Temperature] as 'Two Months Prior: Minimum Hourly Regular Temperature',
b.[Maximum Hourly Bulbo Seco] as 'Two Months Prior: Maximum Hourly Bulbo Seco',
b.[Minimum Hourly Bulbo Seco] as 'Two Months Prior: Minimum Hourly Bulbo Seco',
b.[Standard Deviation Precipitation] as 'Two Months Prior: Standard Deviation Precipitation',
b.[Standard Deviation Bulbo Seco Daily Temperature] as 'Two Months Prior: Standard Deviation Bulbo Seco Daily Temperature',
b.[average temperature bulbo seco] as 'Two Months Prior: Average temperature bulbo seco',
b.[Altitude] as 'Two Months Prior: Altitude',
b.[Regiao] as 'Two Months Prior: Regiao',
b.[UF] as 'Two Months Prior: UF',
b.[Station Latitude] as 'Two Months Prior: Station Latitude',
b.[Station Longitude] as 'Two Months Prior: Station Longitude',
b.[Station Municipality] as 'Two Months Prior: Station Municipality',
b.[Min Distance] as 'Two Months Prior: Min Distance (m)',
b.[Min Distance]/1000.0 as 'Two Months Prior: Min Distance (km)'
into #finaldataset from #onemonthcombination a 
left join #weatherset b on a.[Two Months Prior] = b.[Year.Month] and
convert(decimal(20,15),a.[latitude]) = b.[disaster latitude] and convert(decimal(20,15),a.[longitude]) = b.[disaster longitude] and a.[municipality] = b.[disaster Municipality]
where a.[latitude] is not null and a.[latitude] <> ''

drop table users.sabell_finaldataset 

select * into users.sabell_finaldataset
from #finaldataset

ALTER TABLE users.sabell_finaldataset
drop column "Column 58"

Alter Table users.sabell_Finaldataset
drop column "Column 59"

ALTER Table users.sabell_finaldataset
drop column "Column 60"

select *, [0-3 km] = case when [one month prior: min distance (km)] < 3.0 then 1 else 0 end,
[0-35 km] = case when  [one month prior: min distance (km)] < 35.0 then 1 else 0 end,
[0-85 km] = case when [one month prior: min distance (km)] < 85.0 then 1 else 0 end
From users.sabell_Finaldataset


--Blank Data Assessment


select [Estacao], count(*) as 'Missing Data'
into #Blankdataassessment
from [master].[dbo].[automatic_weather_stations_inmet_brazil_2000_2021]
where [PRECIPITACAO TOTAL HORARIO (mm)] = '' or [PRECIPITACAO TOTAL HORARIO (mm)] = -9999.0
group by [Estacao]

select [Estacao], count(*) as 'Total Data'
into #denominator
from [master].[dbo].[automatic_weather_stations_inmet_brazil_2000_2021]
group by [Estacao]

select a.*, b.[Total Data], ([Missing Data]*1.0)/([Total Data]) as 'Percent Missing' from #Blankdataassessment a
left join #denominator b on a.[estacao] = b.[estacao]


select [Estacao], month([Data (YYYY-MM-DD)]) AS 'MONTH', year([Data (YYYY-MM-DD)]) as 'Year', count(*) as 'Missing Data'
into #Blankdataassessmentbymonth
from [master].[dbo].[automatic_weather_stations_inmet_brazil_2000_2021]
where [PRECIPITACAO TOTAL HORARIO (mm)] = '' or [PRECIPITACAO TOTAL HORARIO (mm)] = -9999.0
group by [Estacao], month([Data (YYYY-MM-DD)]), year([Data (YYYY-MM-DD)])

select [Estacao], month([Data (YYYY-MM-DD)]) AS 'MONTH', year([Data (YYYY-MM-DD)]) as 'Year', count(*) as 'Total Data'
into #denominatorBYMONTH
from [master].[dbo].[automatic_weather_stations_inmet_brazil_2000_2021]
group by [Estacao], month([Data (YYYY-MM-DD)]), year([Data (YYYY-MM-DD)])

select a.*, b.[Total Data], ([Missing Data]*1.0)/([Total Data]) as 'Percent Missing' from #Blankdataassessmentbymonth a
left join #denominatorbymonth b on a.[estacao] = b.[estacao] and a.[month] = b.[month] and a.[year] = b.[year]

