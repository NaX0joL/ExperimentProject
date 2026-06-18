import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from pathlib import Path

from core.experiment import Experiment



class BulkMetricsComputer():
    
    def __init__(self) -> None:
        return
    
    def compute(self, path:Path) -> dict[str, dict]:
        mpkg_paths = PathMapper.scour_for_mpkg(path)
        mpkg_metrics = {}
        
        for path in sorted(mpkg_paths):
            exp = Experiment()
            exp.load(path)
            exp.compute_metrics()
            
            mpkg_metrics[exp.experiment_id] = exp.experiment_state.metrics_result
        
        self.metrics_result = mpkg_metrics
        return mpkg_metrics
    
    def table_print(self) -> None:
        printer = TablePrinter(self.metrics_result)
        printer.show()
        return
    
    def per_metric_print(self) -> None:
        printer = IndividualMetricPrinter(self.metrics_result)
        printer.show()
        return



class PathMapper:
    
    @staticmethod
    def scour_for_mpkg(dir:Path) -> list[Path]:
        dir = Path(dir)
        mpkg_paths = [path.parent for path in dir.rglob("__mpkg__.py")]
        return mpkg_paths



class TablePrinter():
    
    def __init__(self, metrics_result:dict[str, dict[str, dict]], left_offset=1, right_offset=2) -> None:
        self.metrics_result = metrics_result
        self.left_offset = left_offset
        self.right_offset = right_offset
        return
    
    def show(self) -> None:
        column_labels = self._populate_column_labels()
        row_contents = self._build_row_contents(column_labels)
        column_lengths = self._calculate_column_lengths(column_labels, row_contents)
        self._build_table(column_labels, column_lengths, row_contents)
        return
    
    def _populate_column_labels(self) -> list[str]:
        column_labels = ["experiment_id"]
        for metrics in self.metrics_result.values():
            for metric_raws in metrics.values():
                for raw in metric_raws.keys():
                    if not raw in column_labels:
                        column_labels.append(raw)
        return column_labels
    
    def _build_row_contents(self, column_labels:list[str]) -> list[list[any]]:
        row_contents = []
        for exp_id, metrics in self.metrics_result.items():
            row = [exp_id]
            for _, metric_raw in metrics.items():
                for raw_name, raw_value in metric_raw.items():
                    if raw_name in column_labels:
                        row.append(raw_value)
                    else:
                        row.append(None)
            row_contents.append(row)
        return row_contents
    
    def _calculate_column_lengths(self, column_labels:list[str], row_contents:list[list[any]]) -> list[int]:
        column_lengths = []
        for jndex, label in enumerate(column_labels):
            max_length = len(label)
            for index in range(len(row_contents)):
                val = row_contents[index][jndex]
                
                if isinstance(val, float):
                    current_length = len(f"{val:.5f}")
                elif val is not None:
                    current_length = len(str(val))
                else:
                    current_length = 1
                    
                max_length = max(max_length, current_length)
            column_lengths.append(max_length)
        return column_lengths
    
    def _build_table(self, column_labels:list[str], column_lengths:list[int], row_contents:list[list[any]]) -> None:
        self._make_separator_line(column_lengths)
        self._make_header_row(column_labels, column_lengths)
        self._make_separator_line(column_lengths)
        self._fill_table_content(row_contents, column_lengths)
        self._make_separator_line(column_lengths)
        return
    
    def _make_separator_line(self, column_lengths:list[int]) -> None:
        print("+", end="")
        
        for length in column_lengths:
            num = self.left_offset + length + self.right_offset
            print(num * "-" + "+", end="")
        
        print()
        return
    
    def _make_header_row(self, column_labels:list[str], column_lengths:list[int]) -> None:
        print("|", end="")
        
        for label, length in zip(column_labels, column_lengths):
            print(self.left_offset * " ", end="")
            
            print(label, end="")
            print((length - len(label)) * " ", end="")
            
            print(self.right_offset * " ", end="")
            print("|", end="")
        
        print()
        return
    
    def _fill_table_content(self, row_contents:list[list[any]], column_lengths:list[int]) -> None:
        for index in range(len(row_contents)):
            print("|", end="")
            
            for jndex in range(len(row_contents[index])):
                print(self.left_offset * " ", end="")
                
                content = row_contents[index][jndex]
                if isinstance(content, float):
                    content = f"{content:.5f}"
                elif content is not None:
                    content = str(content)
                else:
                    content = "x"
                    
                print(content, end="")
                print((column_lengths[jndex] - len(content)) * " ", end="")
                
                print(self.right_offset * " ", end="")
                print("|", end="")
            
            print()
        return



class IndividualMetricPrinter():
    
    def __init__(self, metrics_result:dict[str, dict]) -> None:
        self.metrics_result = metrics_result
        return
    
    def show(self) -> None:
        metric_names = self._get_metric_name()
        self._print_exp_id()
        self._print_individual_metric(metric_names)
        return
    
    def _get_metric_name(self) -> list[str]:
        metric_names = []
        for metrics in self.metrics_result.values():
            for name, raw in metrics.items():
                if not name in metric_names:
                    metric_names.append(name)
        return metric_names
    
    def _print_exp_id(self) -> None:
        print("Experiment ID:".upper())
        for exp_id in self.metrics_result.keys():
            print(exp_id)
        return
    
    def _print_individual_metric(self, metric_names:list[str]) -> None:
        for name in metric_names:
            print(f"{name}:".upper())
            for _, metrics in self.metrics_result.items():
                print(metrics[name][name])
        return



if __name__ == "__main__":
    path = "savefolder/mpkg/test"
    computer = BulkMetricsComputer()
    computer.compute(path=path)
    computer.table_print()
    computer.per_metric_print()
    print("DONE!")