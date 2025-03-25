import json
from veil_processor import process_data

def handle_request(request_data):
    try:
        # Parse the request
        data = json.loads(request_data)
        input1 = int(data.get('input1', 0))
        input2 = int(data.get('input2', 0))
        mode = data.get('mode', 'single')  # 'single' or 'multiple'
        
        # Process the data using the veil algorithm
        if mode == 'single':
            result = process_data(input1, input2, mode='single')
            response = {'result': result}
        else:  # multiple samples
            stats = process_data(input1, input2, mode='multiple')
            
            response = {
                'results_summary': {
                    'total_samples': stats['total_samples'],
                    'first_value': stats['output_values'][0],
                    'second_value': stats['output_values'][1],
                    'count_first': stats['counts']['first'],
                    'count_second': stats['counts']['second'],
                    'ratio_second': stats['overall_ratio'],
                    'grid_stats': stats['grid_stats'],
                    'raw_runs': stats['all_runs']
                }
            }
            
        return json.dumps(response)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return json.dumps({'error': str(e), 'details': error_details}) 