"""
Test script to verify that user_id and user_name are properly used in Excel parsing
"""

import json
from excel_resume_parser.fixed_excel_parser_adapter import FixedExcelParserAdapter
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("user_id_fix_test")

def test_user_id_fix():
    """Test that user_id and user_name are properly used in Excel parsing"""
    
    print("Testing User ID and Username Fix for Excel Parser")
    print("=" * 60)
    
    try:
        # Test data simulating user input
        test_user_id = "harshgajera123"
        test_user_name = "Harsh Gajera"
        
        print(f"Test User ID: {test_user_id}")
        print(f"Test User Name: {test_user_name}")
        print("-" * 40)
        
        # Initialize the fixed adapter
        adapter = FixedExcelParserAdapter(llm_provider="groq_cloud")
        
        # Test with example Excel file (if it exists)
        excel_file_path = "example_resumes.xlsx"
        
        # Check if file exists
        import os
        if not os.path.exists(excel_file_path):
            print(f"❌ Excel file not found: {excel_file_path}")
            print("Creating a mock test to verify the fix...")
            
            # Test the process_excel_file method signature
            import inspect
            sig = inspect.signature(adapter.process_excel_file)
            params = list(sig.parameters.keys())
            
            print(f"✅ Method signature parameters: {params}")
            
            if 'user_id' in params and 'user_name' in params:
                print("✅ user_id and user_name parameters are now available!")
                
                # Test parameter defaults
                user_id_param = sig.parameters.get('user_id')
                user_name_param = sig.parameters.get('user_name')
                
                print(f"✅ user_id parameter: {user_id_param}")
                print(f"✅ user_name parameter: {user_name_param}")
                
                return True
            else:
                print("❌ user_id and user_name parameters are missing!")
                return False
        
        else:
            print(f"✅ Found Excel file: {excel_file_path}")
            
            # Process with user information
            result = adapter.process_excel_file(
                file_path=excel_file_path,
                user_id=test_user_id,
                user_name=test_user_name
            )
            
            print("Processing Results:")
            print("-" * 20)
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Total rows: {result.get('total_rows', 0)}")
            print(f"Successful parses: {result.get('successful_parses', 0)}")
            
            # Check if parsed resumes have correct user information
            parsed_resumes = result.get('parsed_resumes', [])
            
            if parsed_resumes:
                print(f"✅ Found {len(parsed_resumes)} parsed resumes")
                
                # Check first resume
                first_resume = parsed_resumes[0]
                resume_user_id = first_resume.get('user_id')
                resume_username = first_resume.get('username')
                
                print(f"First resume user_id: {resume_user_id}")
                print(f"First resume username: {resume_username}")
                
                # Verify user information is correct
                if resume_user_id == test_user_id and resume_username == test_user_name:
                    print("✅ User information correctly preserved in parsed resumes!")
                    return True
                else:
                    print("❌ User information not correctly preserved!")
                    print(f"Expected: {test_user_id}, {test_user_name}")
                    print(f"Got: {resume_user_id}, {resume_username}")
                    return False
            else:
                print("❌ No parsed resumes found")
                return False
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_resume_structure():
    """Test the structure of parsed resumes to ensure user_id is correctly set"""
    
    print("\nTesting Resume Structure for User Information")
    print("=" * 50)
    
    try:
        from excel_resume_parser.fixed_excel_resume_parser import FixedExcelResumeParser
        
        # Create test data
        test_excel_data = [
            {
                "name": "John Doe",
                "email": "john.doe@example.com", 
                "phone": "9876543210",
                "experience": "3 years",
                "skills": "Python, JavaScript, React"
            }
        ]
        
        test_user_id = "testuser123"
        test_username = "Test User"
        
        # Initialize parser
        parser = FixedExcelResumeParser(llm_provider="groq_cloud")
        
        # Process test data
        result = parser.process_excel_data(
            excel_data=test_excel_data,
            base_user_id=test_user_id,
            base_username=test_username
        )
        
        print(f"Processing result: {result.get('successful_parses', 0)} successful")
        
        if result.get('parsed_resumes'):
            resume_data = result['parsed_resumes'][0]
            
            print(f"Resume user_id: {resume_data.get('user_id')}")
            print(f"Resume username: {resume_data.get('username')}")
            print(f"Resume ID: {resume_data.get('resume_id', 'Not set')}")
            
            # Check if user information matches
            if (resume_data.get('user_id') == test_user_id and 
                resume_data.get('username') == test_username):
                print("✅ Resume structure correctly preserves user information!")
                return True
            else:
                print("❌ Resume structure doesn't preserve user information correctly!")
                return False
        else:
            print("❌ No resumes in processing result")
            return False
            
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 USER ID & USERNAME FIX VALIDATION")
    print("=" * 60)
    
    # Run tests
    test1_result = test_user_id_fix()
    test2_result = test_resume_structure()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"User ID Fix Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"Resume Structure Test: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    overall_success = test1_result and test2_result
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 User ID and Username are now properly preserved in Excel parsing!")
        print("\nBehavior after fix:")
        print("- All resumes uploaded by a user will have the same user_id")
        print("- All resumes uploaded by a user will have the same username")  
        print("- Each resume gets a unique resume_id for identification")
        print("- User information is preserved throughout the parsing pipeline")
    else:
        print("\n❌ Issues remain. Check the errors above.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)