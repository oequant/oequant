import pandas as pd

def calculate_returns0(df, periods=(1, 2, 3, 5, 10), forward=False, per_period=False, calculate_ons_ids=True,
                       price_column=None,  # default is Close
                       price_column_entry=None,  # default is Close as well (set e.g. to Open to use different price for entry)
                       join_with_df=False,
                       ):
    rets = pd.DataFrame(index=df.index)
    
    if calculate_ons_ids:
        open_col_name = 'Open' if 'Open' in df.columns else 'open'
        close_col_name = 'Close' if 'Close' in df.columns else 'close'
        if forward:
            rets["forward_return_c2o"] = ((df[open_col_name] - df[close_col_name].shift(1)) / df[close_col_name].shift(1)).shift(-1)
            rets["forward_return_o2c"] = ((df[close_col_name] - df[open_col_name]) / df[open_col_name]).shift(-1)
        else:
            rets["return_c2o"] = (df[open_col_name] - df[close_col_name].shift(1)) / df[close_col_name].shift(1)
            rets["return_o2c"] = (df[close_col_name] - df[open_col_name]) / df[open_col_name]

    if price_column is None:
        price_column = 'Close' if 'Close' in df.columns else 'close'
    if price_column_entry is None:
        price_column_entry = price_column
        
    for period in periods:
        period_str = str(period).zfill(2)
        if forward:
            ret = df[price_column].shift(-period) / df[price_column_entry] - 1
            if per_period:
                ret = ret / period
            rets[f"forward_return_{period_str}"] = ret
        else:
            ret = df[price_column] / df[price_column_entry].shift(period) - 1
            if per_period:
                ret = ret / period
            rets[f"return_{period_str}"] = ret
            
    if join_with_df:
        rets = pd.concat([df, rets], axis=1)
    return rets

def calculate_returns(df, **kwargs):
    if len(df.index.names) == 1:
        return calculate_returns0(df, **kwargs)
    elif len(df.index.names) == 2:
        original_index_names = list(df.index.names) # e.g., ['date', 'symbol']
        grouping_level_name = original_index_names[1] # 'symbol'
        inner_index_name = original_index_names[0]    # 'date'
        
        # Use groupby().apply()
        grouped_result = df.groupby(level=grouping_level_name).apply(lambda x: calculate_returns0(x, **kwargs))
        
        # Check if the result has 3 levels, which was the problematic case observed
        if grouped_result.index.nlevels == 3:
            # Assuming names are [grouping_level_name, inner_index_name, grouping_level_name_again]
            # e.g., ['symbol', 'date', 'symbol']
            # We want to drop the innermost (third) level.
            # The specific name of the third level might be grouping_level_name or it might be None
            # if calculate_returns0 returned an unnamed index that then got a duplicate name.
            # Safest to drop by position if it's consistently the third level that's extraneous.
            try:
                grouped_result = grouped_result.reset_index(level=2, drop=True)
                # After reset, the names should be [grouping_level_name, inner_index_name]
                grouped_result.index.names = [grouping_level_name, inner_index_name]
            except Exception as e_reset:
                print(f"Warning: Failed to fix 3-level index from groupby.apply. Index: {grouped_result.index.names}, Error: {e_reset}")
        elif grouped_result.index.nlevels == 2:
            # If 2 levels, ensure names are correct: [grouping_level_name, inner_index_name]
            expected_names = [grouping_level_name, inner_index_name]
            if list(grouped_result.index.names) != expected_names:
                try:
                    grouped_result.index.names = expected_names
                except Exception as e_rename:
                     print(f"Warning: Could not set expected 2-level index names '{expected_names}'. Current: {list(grouped_result.index.names)}. Error: {e_rename}")       
        else:
            # Unexpected number of levels
            print(f"Warning: Grouped result has unexpected number of levels: {grouped_result.index.nlevels}. Names: {list(grouped_result.index.names)}")

        return grouped_result
    else:
        assert False, "DataFrame index must have 1 or 2 levels." 