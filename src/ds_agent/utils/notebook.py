import os
import nbformat
from ds_agent.core.state import AgentState

def save_session_to_ipynb(state: AgentState, filename: str = 'analysis.ipynb') -> str:
    """
    Exports the current agent state to a standard Jupyter Notebook (.ipynb) file.
    """
    nb = nbformat.v4.new_notebook()
    cells = []

    # Iterate through the tracked notebook cells
    for cell_data in state.get('notebook_cells', []):
        cell_type = cell_data.get('cell_type')
        source = cell_data.get('source', '')
        execution_count = cell_data.get('execution_count', None)
        
        if cell_type == 'markdown':
            nb_cell = nbformat.v4.new_markdown_cell(source=source)
            cells.append(nb_cell)
            
        elif cell_type == 'code':
            outputs = []
            for output in cell_data.get('outputs', []):
                output_type = output.get('type')
                
                if output_type == 'stdout' or output_type == 'stderr':
                    nb_output = nbformat.v4.new_output(
                        output_type='stream',
                        name=output_type,
                        text=output.get('text', '')
                    )
                    outputs.append(nb_output)
                    
                elif output_type == 'image':
                    image_data = output.get('data')
                    mime_type = output.get('mime_type', 'image/png')
                    nb_output = nbformat.v4.new_output(
                        output_type='display_data',
                        data={mime_type: image_data}
                    )
                    outputs.append(nb_output)
                
                elif output_type == 'error':
                    nb_output = nbformat.v4.new_output(
                        output_type='error',
                        ename=output.get('ename', 'Error'),
                        evalue=output.get('evalue', ''),
                        traceback=output.get('traceback', [])
                    )
                    outputs.append(nb_output)
                
                elif output_type == 'result':
                    nb_output = nbformat.v4.new_output(
                        output_type='execute_result',
                        execution_count=execution_count,
                        data=output.get('data', {}),
                        metadata={}
                    )
                    outputs.append(nb_output)

            nb_cell = nbformat.v4.new_code_cell(source=source, execution_count=execution_count)
            nb_cell.outputs = outputs
            cells.append(nb_cell)

    nb.cells = cells
    
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    return filename