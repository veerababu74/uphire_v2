# Skills Recommendation API

This module provides intelligent skills recommendations based on job positions using data from the MongoDB database. It analyzes resumes and skills data to suggest relevant skills for different job roles.

## Features

### 1. Position-Based Skills Recommendations
Get skills recommendations for specific job positions like "Python Developer", "Backend Developer", etc.

**Endpoint:** `GET /recommendations/skills/{position}`

**Examples:**
- `/recommendations/skills/Python Developer`
- `/recommendations/skills/Backend Developer`
- `/recommendations/skills/Full Stack Developer`
- `/recommendations/skills/Data Scientist`
- `/recommendations/skills/DevOps Engineer`

**Response:**
```json
{
  "success": true,
  "data": {
    "position": "Python Developer",
    "total_skills_found": 15,
    "resumes_analyzed": 45,
    "recommended_skills": [
      {
        "skill": "Python",
        "relevance_score": 12.5,
        "frequency": 12
      },
      {
        "skill": "Django",
        "relevance_score": 8.2,
        "frequency": 8
      },
      {
        "skill": "Flask",
        "relevance_score": 6.7,
        "frequency": 6
      }
    ],
    "search_keywords": ["python", "django", "flask", "fastapi"]
  },
  "message": "Successfully generated skills recommendations for Python Developer"
}
```

### 2. Skills Search by Keyword
Search for skills containing specific keywords.

**Endpoint:** `GET /recommendations/skills/search/{keyword}`

**Examples:**
- `/recommendations/skills/search/java`
- `/recommendations/skills/search/react`
- `/recommendations/skills/search/machine`

**Response:**
```json
{
  "success": true,
  "keyword": "java",
  "total_found": 8,
  "skills": [
    "Java",
    "JavaScript",
    "Java Spring",
    "Java Hibernate"
  ]
}
```

### 3. Popular Job Positions
Get the most popular job positions based on resume data.

**Endpoint:** `GET /recommendations/positions/popular`

**Response:**
```json
{
  "success": true,
  "total_positions": 20,
  "popular_positions": [
    {
      "position": "Software Engineer",
      "frequency": 125
    },
    {
      "position": "Python Developer",
      "frequency": 89
    },
    {
      "position": "Full Stack Developer",
      "frequency": 67
    }
  ]
}
```

### 4. Trending Skills
Get trending skills based on frequency across all resumes.

**Endpoint:** `GET /recommendations/skills/trending`

**Response:**
```json
{
  "success": true,
  "total_skills": 30,
  "trending_skills": [
    {
      "skill": "Python",
      "frequency": 456
    },
    {
      "skill": "JavaScript",
      "frequency": 389
    },
    {
      "skill": "React",
      "frequency": 267
    }
  ]
}
```

## How It Works

### 1. Position Analysis
The system analyzes job positions by:
- Extracting keywords from position names
- Mapping positions to relevant technology stacks
- Searching through resume data for matching patterns

### 2. Skills Extraction
Skills are gathered from multiple sources:
- **Resume Skills Fields:** Direct skills listed in resumes
- **Experience Descriptions:** Technologies mentioned in job descriptions
- **Skills Database:** Curated skills from the skills_titles collection
- **Job Titles:** Technologies inferred from job titles

### 3. Relevance Scoring
Skills are scored based on:
- **Frequency:** How often the skill appears
- **Context:** Where the skill was found (direct skills vs descriptions)
- **Co-occurrence:** Skills that appear together with position keywords

### 4. Position Keyword Mapping
The system includes intelligent mapping for common positions:

```python
{
    "python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
    "java": ["java", "spring", "hibernate", "maven", "gradle"],
    "javascript": ["javascript", "node", "react", "angular", "vue"],
    "frontend": ["html", "css", "javascript", "react", "angular", "vue"],
    "backend": ["python", "java", "node", "express", "django", "spring"],
    "fullstack": ["javascript", "python", "react", "node", "django", "flask"],
    "data": ["python", "sql", "pandas", "numpy", "machine learning", "tensorflow"],
    "devops": ["docker", "kubernetes", "aws", "jenkins", "terraform"],
    "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
    "ai": ["machine learning", "deep learning", "tensorflow", "pytorch", "nlp"]
}
```

## API Parameters

### Common Parameters
- `limit`: Number of results to return (default: 20, max: 100)

### Skills Recommendation Parameters
- `position`: Job position (required)
- `limit`: Number of skills to recommend (1-100)

### Search Parameters
- `keyword`: Search keyword (required)
- `limit`: Number of results (1-50)

## Usage Examples

### 1. Get Python Developer Skills
```bash
curl -X GET "http://localhost:8000/recommendations/skills/Python%20Developer?limit=10"
```

### 2. Search for Cloud Skills
```bash
curl -X GET "http://localhost:8000/recommendations/skills/search/cloud?limit=15"
```

### 3. Get Popular Positions
```bash
curl -X GET "http://localhost:8000/recommendations/positions/popular?limit=20"
```

### 4. Get Trending Skills
```bash
curl -X GET "http://localhost:8000/recommendations/skills/trending?limit=25"
```

## Integration

The recommendations API integrates with:
- **MongoDB Resume Collection:** Main source of skills data
- **Skills Titles Collection:** Curated skills database
- **Experience Data:** Job descriptions and titles
- **Vector Search:** For enhanced similarity matching

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request:** Invalid parameters
- **404 Not Found:** No data found for position
- **500 Internal Server Error:** Database or processing errors

## Performance Considerations

- **Caching:** Results can be cached for frequently requested positions
- **Batch Processing:** Large datasets are processed in batches
- **Indexing:** MongoDB indexes on skills and experience fields
- **Limits:** Built-in limits prevent excessive resource usage

## Future Enhancements

1. **Machine Learning:** Use ML models for better skill relevance scoring
2. **Caching:** Implement Redis caching for popular positions
3. **Real-time Updates:** Skills recommendations that update with new resume data
4. **Skill Categories:** Group skills by categories (programming languages, frameworks, etc.)
5. **Skill Relationships:** Identify related and complementary skills
6. **Industry-specific:** Position recommendations by industry verticals

## Data Sources

The recommendations are based on:
- **Resume Database:** Active resume collection with 10,000+ resumes
- **Skills Collection:** Curated list of technology skills and job titles
- **Experience Data:** Job descriptions, titles, and company information
- **Real-time Data:** Updated as new resumes are added to the system
