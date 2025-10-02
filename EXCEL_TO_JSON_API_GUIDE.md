# Excel to JSON Conversion API

## Overview

The Excel to JSON Conversion API provides a powerful way to convert Excel files into properly cleaned JSON format without performing resume parsing. This is ideal when you need to:

- Clean and standardize Excel data
- Convert Excel to JSON for further processing
- Remove empty rows, invalid values, and normalize data
- Handle mixed data types properly

## API Endpoint

```
POST /resume-parser/excel-to-json
```

## Features

✅ **Data Cleaning**: Removes NaN, empty, and invalid values  
✅ **Type Conversion**: Converts numeric strings to proper numbers  
✅ **Header Normalization**: Optional lowercase headers with underscores  
✅ **Empty Row Filtering**: Skip completely empty rows  
✅ **Duplicate Header Handling**: Intelligent handling of duplicate columns  
✅ **Row Limiting**: Process only specified number of rows  
✅ **Multiple Excel Formats**: Supports .xlsx, .xls, .xlsm  

## Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Excel file to convert (.xlsx, .xls, .xlsm) |
| `sheet_name` | String | No | First sheet | Name or index of sheet to process |
| `skip_empty_rows` | Boolean | No | `true` | Skip completely empty rows |
| `normalize_headers` | Boolean | No | `true` | Convert headers to lowercase with underscores |
| `max_rows` | Integer | No | All rows | Maximum number of rows to process |

## Response Structure

```json
{
  "status": "success",
  "message": "Excel file successfully converted to clean JSON",
  "filename": "data.xlsx",
  "sheet_name": null,
  "statistics": {
    "original_rows": 100,
    "cleaned_rows": 95,
    "rows_removed": 5,
    "processing_time_seconds": 1.23
  },
  "settings": {
    "skip_empty_rows": true,
    "normalize_headers": true,
    "max_rows": null
  },
  "data": [
    // Complete array of cleaned JSON objects
    {
      "name": "John Doe",
      "email": "john@email.com",
      "age": 25,
      "salary": 50000
    }
    // ... more records
  ]
}
```

## Usage Examples

### Python Example

```python
import requests

# Basic conversion
url = "http://localhost:8000/resume-parser/excel-to-json"

with open("data.xlsx", "rb") as f:
    files = {"file": ("data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {
        "skip_empty_rows": True,
        "normalize_headers": True
    }
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        cleaned_data = result["data"]
        print(f"Converted {len(cleaned_data)} rows")
        
        # Use the cleaned data
        for row in cleaned_data:
            print(row)
```

### JavaScript/Fetch Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('skip_empty_rows', 'true');
formData.append('normalize_headers', 'true');

fetch('/resume-parser/excel-to-json', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Conversion successful:', data);
    const cleanedData = data.data;
    // Process cleaned data
});
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/resume-parser/excel-to-json" \
  -F "file=@data.xlsx" \
  -F "skip_empty_rows=true" \
  -F "normalize_headers=true" \
  -F "max_rows=100"
```

## Data Cleaning Process

### 1. **NaN and Empty Value Handling**
- Converts pandas NaN to `null`
- Handles `None`, `"N/A"`, `"NA"`, `"#N/A"` strings
- Removes empty strings and whitespace-only values

### 2. **Type Conversion**
- Numeric strings → Numbers (int/float)
- Boolean-like strings → Booleans (where appropriate)
- Preserves original strings when conversion isn't possible

### 3. **Header Normalization** (Optional)
```
"First Name" → "first_name"
"E-Mail Address" → "e_mail_address" 
"Phone # " → "phone"
```

### 4. **Row Filtering**
- Skips rows where all values are empty/null
- Preserves rows with at least one valid value

## Error Handling

### Invalid File Type
```json
{
  "detail": "Invalid file type. Only Excel files (.xlsx, .xls, .xlsm) are allowed."
}
```

### Empty File
```json
{
  "status": "warning",
  "message": "No data found in Excel file",
  "data": []
}
```

### Processing Error
```json
{
  "detail": "Excel to JSON conversion failed: [error details]"
}
```

## Integration Workflow

### 1. **Upload and Convert**
```python
# Convert Excel to clean JSON
response = requests.post(url, files=files, data=params)
clean_data = response.json()["data"]
```

### 2. **Process Cleaned Data**
```python
for row in clean_data:
    # Data is already cleaned and typed properly
    name = row.get("name")  # string or None
    age = row.get("age")    # int or None
    salary = row.get("salary")  # float/int or None
```

### 3. **Further Processing**
```python
# Save to database
for row in clean_data:
    db.insert(row)

# Convert to pandas DataFrame
df = pd.DataFrame(clean_data)

# Export to other formats
with open("cleaned_data.json", "w") as f:
    json.dump(clean_data, f, indent=2)
```

## Performance Tips

1. **Batch Processing**: For large files, use `max_rows` to process in chunks
2. **Memory Management**: Process files < 50MB for optimal performance
3. **Header Normalization**: Enable for consistent field names across files
4. **Empty Row Skipping**: Always enable to reduce data size

## Common Use Cases

### 1. **Data Migration**
Convert legacy Excel files to JSON for modern applications.

### 2. **ETL Pipelines**
Clean and standardize Excel data before database insertion.

### 3. **API Data Preparation**
Convert uploaded Excel files to API-ready JSON format.

### 4. **Data Analytics**
Clean Excel data for analysis and visualization tools.

## Testing

Run the included test script to verify the API:

```bash
python test_excel_to_json_api.py
```

This will test:
- Basic conversion functionality
- Header normalization options
- Row limiting features
- Error handling for invalid files

## Next Steps

After getting clean JSON data, you can:

1. **Save to Database**: Insert the cleaned records
2. **Further Processing**: Apply business logic to clean data
3. **Resume Parsing**: Use the clean data with resume parsing APIs
4. **Data Analysis**: Generate reports and insights
5. **API Integration**: Send clean data to other services

The clean JSON format ensures consistent, properly-typed data that's ready for any downstream processing.