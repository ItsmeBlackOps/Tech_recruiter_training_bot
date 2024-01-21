---

**1. REST API: Highest International Students**

Use the HTTP GET method to retrieve information from a database of universities across the globe. Query `https://jsonmock.hackerrank.com/api/universities` to find all the records. The query result is paginated and can be accessed by appending to the query string `/page=num` where num is the page number.

The response is a JSON object with the following 5 fields:

- `page`: The current number of the results. (Number)
- `per_page`: The total number of results returned per page. (Number)
- `total`: The total number of results. (Number)
- `total_pages`: The total number of pages with results. (Number)
- `data`: Either an empty array or an array with a single object that contains the universities' records.

Example of a data array object:

```
{
  "university": "King's College London",
  "rank_display": "35",
  "score": 82,
  "type": "Public",
  "student_faculty_ratio": 7,
  "international_students": "15,075",
  "faculty_count": "4,216",
  "location": {
    "city": "London",
    "country": "United Kingdom",
    "region": "Europe"
  }
}
```

In `data`, each university has the following schema:

- `university`: The name of the university (String)
- `rank_display`: The rank of the university according to the 2022 QS Rankings (String).
- `score`: The score of the university according to the 2022 QS Rankings (Number).
- `type`: The type of university (String)
- `student_faculty_ratio`: The ratio of number of students to the number of faculty. (Number)
- `international_students`: The number of international students (String).
- `faculty_count`: The number of faculty (String)
- `location`: An object containing the location details. The object has the following schema:
  - `city`: (String)
  - `country`: (String)
  - `region`: (String)

Complete the `highestInternationalStudents` function where given the name of two cities as parameters, return the name of the university with the highest number of international students in the first city. If the first city does not have a university within the data, return the university with the highest number of international students in the second city.

**Function Description**

Complete the function `highestInternationalStudents` in the editor.

`highestInternationalStudents` has the following parameters:

- string `firstCity`: name of the first city
- string `secondCity`: name of the second city

**Returns**

- string: the university with the highest number of international students.

**Constraints**

There is always a university in one of the two cities.

**Note**: Please review the header in the code stub to see available libraries for API requests in the selected language. Required libraries can be imported in order to solve the question. Check our full list of supported libraries at `https://www.hackerrank.com/environment`.

**Sample Case 0**

Sample Input For Custom Testing

```
Singapore
New York City
```

Sample Output

```
National University of Singapore (NUS)
```
---