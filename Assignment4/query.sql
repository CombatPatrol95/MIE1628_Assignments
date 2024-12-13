-- Filter all the OCCUPATIONS in MAJOR_CATEGORY of Computer, Engineering, and Science for the YEAR 2013
SELECT occupation 
FROM dbo.EmploymentData
WHERE major_category = 'Computer, Engineering, and Science' AND year = 2013;

-- How many OCCUPATIONS exist in the MINOR_CATEGORY of Business and Financial Operations overall?
SELECT COUNT(DISTINCT occupation) 
FROM dbo.EmploymentData
WHERE minor_category = 'Business and Financial Operations';

-- Get all relevant information for bus drivers across all years
SELECT * 
FROM dbo.EmploymentData
WHERE occupation = 'Bus drivers';

-- Summarize the total number of WORKERS_FEMALE in the MAJOR_CATEGORY of Management, Business, and Financial by each year
SELECT year, SUM(CAST(workers_female AS INT)) AS total_female_workers
FROM dbo.EmploymentData
WHERE major_category = 'Management, Business, and Financial'
GROUP BY year;

-- What were the total earnings of male (TOTAL_EARNINGS_MALE) employees in the Service MAJOR_CATEGORY for the year 2015?
SELECT SUM(CAST(total_earnings_male AS INT)) AS total_male_earnings 
FROM dbo.EmploymentData
WHERE major_category = 'Service' AND year = 2015;

-- Compare the TOTAL_EARNINGS_MALE and TOTAL_EARNINGS_FEMALE earnings irrespective of occupation by each year.
SELECT 
    year, 
    SUM(CASE WHEN ISNUMERIC(total_earnings_male) = 1 THEN CAST(total_earnings_male AS DECIMAL(18,2)) ELSE 0 END) AS total_male_earnings, 
    SUM(CASE WHEN ISNUMERIC(total_earnings_female) = 1 THEN CAST(total_earnings_female AS DECIMAL(18,2)) ELSE 0 END) AS total_female_earnings
FROM dbo.EmploymentData
GROUP BY year;

-- How much money (TOTAL_EARNINGS_FEMALE) did female workers make as engineers in 2016?
SELECT SUM(CASE WHEN ISNUMERIC(total_earnings_female) = 1 THEN CAST(total_earnings_female AS DECIMAL(18,2)) ELSE 0 END) 
FROM dbo.EmploymentData
WHERE occupation = 'Engineers' AND year = 2016;

-- What is the total number of full-time and part-time female workers versus male workers year over year?
SELECT 
    year,
    SUM(CASE WHEN ISNUMERIC(full_time_female) = 1 THEN CAST(full_time_female AS DECIMAL(18,2)) ELSE 0 END) AS total_full_time_female,
    SUM(CASE WHEN ISNUMERIC(part_time_female) = 1 THEN CAST(part_time_female AS DECIMAL(18,2)) ELSE 0 END) AS total_part_time_female,
    SUM(CASE WHEN ISNUMERIC(full_time_male) = 1 THEN CAST(full_time_male AS DECIMAL(18,2)) ELSE 0 END) AS total_full_time_male,
    SUM(CASE WHEN ISNUMERIC(part_time_male) = 1 THEN CAST(part_time_male AS DECIMAL(18,2)) ELSE 0 END) AS total_part_time_male
FROM dbo.EmploymentData
GROUP BY year;