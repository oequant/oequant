import pandas as pd
import numpy as np
import pytest
from oequant.research.returns import calculate_returns

@pytest.fixture
def sample_data_single_index():
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    }
    dates = pd.date_range(start='2023-01-01', periods=11)
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def sample_data_multi_index():
    dates = pd.date_range(start='2023-01-01', periods=5)
    symbols = ['AAPL', 'GOOG']
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    data = {
        'Open': [100, 200, 101, 201, 102, 202, 103, 203, 104, 204],
        'Close': [101, 201, 102, 202, 103, 203, 104, 204, 105, 205],
        'High': [102, 202, 103, 203, 104, 204, 105, 205, 106, 206],
        'Low': [99, 199, 100, 200, 101, 201, 102, 202, 103, 203]
    }
    df = pd.DataFrame(data, index=index)
    return df

def test_calculate_returns_single_index_basic(sample_data_single_index):
    df = sample_data_single_index
    periods = (1, 5)
    result_df = calculate_returns(df.copy(), periods=periods, forward=False, calculate_ons_ids=True)
    
    assert "return_01" in result_df.columns
    assert "return_05" in result_df.columns
    assert "return_c2o" in result_df.columns
    assert "return_o2c" in result_df.columns
    
    # Check for NaNs at the beginning due to shift
    assert pd.isna(result_df['return_01'].iloc[0])
    assert pd.isna(result_df['return_05'].iloc[4]) # period is 5, so first 5 will be NaN
    assert pd.isna(result_df['return_c2o'].iloc[0])

    # Check a known value for return_01 (Close[1]/Close[0] - 1)
    expected_ret_01_idx_1 = (df['Close'].iloc[1] / df['Close'].iloc[0]) - 1
    assert np.isclose(result_df['return_01'].iloc[1], expected_ret_01_idx_1)

    # Check overnight return c2o: (Open[1] - Close[0]) / Close[0]
    expected_c2o_idx_1 = (df['Open'].iloc[1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    assert np.isclose(result_df['return_c2o'].iloc[1], expected_c2o_idx_1)

    # Check intraday return o2c: (Close[0] - Open[0]) / Open[0]
    expected_o2c_idx_0 = (df['Close'].iloc[0] - df['Open'].iloc[0]) / df['Open'].iloc[0]
    assert np.isclose(result_df['return_o2c'].iloc[0], expected_o2c_idx_0)


def test_calculate_returns_single_index_forward(sample_data_single_index):
    df = sample_data_single_index
    periods = (1, 3)
    result_df = calculate_returns(df.copy(), periods=periods, forward=True, calculate_ons_ids=True)

    assert "forward_return_01" in result_df.columns
    assert "forward_return_03" in result_df.columns
    assert "forward_return_c2o" in result_df.columns
    assert "forward_return_o2c" in result_df.columns

    # Check for NaNs at the end due to forward shift
    assert pd.isna(result_df['forward_return_01'].iloc[-1])
    assert pd.isna(result_df['forward_return_03'].iloc[-3]) # period is 3, so last 3 will be NaN
    assert pd.isna(result_df['forward_return_c2o'].iloc[-1])
    assert pd.isna(result_df['forward_return_o2c'].iloc[-1])
    
    # Check a known value for forward_return_01 (Close[1]/Close[0] - 1)
    # Forward return at index 0 using period 1: (df['Close'].shift(-1) / df['Close']) - 1
    expected_fwd_ret_01_idx_0 = (df['Close'].iloc[1] / df['Close'].iloc[0]) - 1
    assert np.isclose(result_df['forward_return_01'].iloc[0], expected_fwd_ret_01_idx_0)
    
    # Check forward overnight return c2o at index 0: ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(-1)
    # At index 0, this is (Open[1] - Close[0]) / Close[0]
    expected_fwd_c2o_idx_0 = (df['Open'].iloc[1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    assert np.isclose(result_df['forward_return_c2o'].iloc[0], expected_fwd_c2o_idx_0)

    # Check forward intraday return o2c at index 0: ((df['Close'] - df['Open']) / df['Open']).shift(-1)
    # At index 0, this is (Close[1] - Open[1]) / Open[1]
    expected_fwd_o2c_idx_0 = (df['Close'].iloc[1] - df['Open'].iloc[1]) / df['Open'].iloc[1]
    assert np.isclose(result_df['forward_return_o2c'].iloc[0], expected_fwd_o2c_idx_0)

def test_calculate_returns_multi_index_basic(sample_data_multi_index):
    df = sample_data_multi_index
    periods = (1, 2)
    result_df = calculate_returns(df.copy(), periods=periods, forward=False, calculate_ons_ids=True)

    assert "return_01" in result_df.columns
    assert "return_02" in result_df.columns
    
    # Check for AAPL
    aapl_df = df.xs('AAPL', level='symbol')
    aapl_result_df = result_df.xs('AAPL', level='symbol')
    
    assert pd.isna(aapl_result_df['return_01'].iloc[0])
    expected_ret_01_idx_1_aapl = (aapl_df['Close'].iloc[1] / aapl_df['Close'].iloc[0]) - 1
    assert np.isclose(aapl_result_df['return_01'].iloc[1], expected_ret_01_idx_1_aapl)

    # Check for GOOG
    goog_df = df.xs('GOOG', level='symbol')
    goog_result_df = result_df.xs('GOOG', level='symbol')
    
    assert pd.isna(goog_result_df['return_01'].iloc[0])
    expected_ret_01_idx_1_goog = (goog_df['Close'].iloc[1] / goog_df['Close'].iloc[0]) - 1
    assert np.isclose(goog_result_df['return_01'].iloc[1], expected_ret_01_idx_1_goog)

def test_calculate_returns_options(sample_data_single_index):
    df = sample_data_single_index.copy()
    
    # Test: no ONS IDs
    res_no_ons = calculate_returns(df, calculate_ons_ids=False, periods=(1,))
    assert "return_c2o" not in res_no_ons.columns
    assert "return_o2c" not in res_no_ons.columns
    assert "return_01" in res_no_ons.columns

    # Test: per_period
    res_per_period = calculate_returns(df, periods=(5,), per_period=True, calculate_ons_ids=False)
    expected_ret_05_idx_5_raw = (df['Close'].iloc[5] / df['Close'].iloc[0]) - 1
    assert np.isclose(res_per_period['return_05'].iloc[5], expected_ret_05_idx_5_raw / 5)

    # Test: join_with_df
    res_joined = calculate_returns(df, periods=(1,), join_with_df=True, calculate_ons_ids=False)
    assert "Close" in res_joined.columns # Original column
    assert "return_01" in res_joined.columns # Calculated column
    assert res_joined.shape[1] == df.shape[1] + 1 # df cols + 1 return col

    # Test: custom price_column and price_column_entry
    df['CustomPrice'] = df['Close'] * 1.1
    df['EntryPrice'] = df['Open'] * 0.9
    res_custom_price = calculate_returns(df, periods=(1,), price_column='CustomPrice', price_column_entry='EntryPrice', calculate_ons_ids=False)
    expected_custom_ret_idx_1 = (df['CustomPrice'].iloc[1] / df['EntryPrice'].iloc[0]) -1 # This is forward=False with period 1
    # Correction: for forward=False, it's current CustomPrice / previous EntryPrice shifted by period
    # So, for period 1 at index 1: df['CustomPrice'].iloc[1] / df['EntryPrice'].iloc[0] - 1
    # No, that's not right. Default logic for non-forward is:
    # ret = df[price_column] / df[price_column_entry].shift(period) - 1
    # So for period=1, at index 1: df['CustomPrice'].iloc[1] / df['EntryPrice'].shift(1).iloc[1] -1
    # which is df['CustomPrice'].iloc[1] / df['EntryPrice'].iloc[0] -1. Yes, this seems correct.
    assert np.isclose(res_custom_price['return_01'].iloc[1], (df['CustomPrice'].iloc[1] / df['EntryPrice'].iloc[0]) - 1)

    res_custom_price_fwd = calculate_returns(df, periods=(1,), price_column='CustomPrice', price_column_entry='EntryPrice', forward=True, calculate_ons_ids=False)
    # Forward logic: df[price_column].shift(-period) / df[price_column_entry] - 1
    # For period=1 at index 0: df['CustomPrice'].shift(-1).iloc[0] / df['EntryPrice'].iloc[0] -1
    # which is df['CustomPrice'].iloc[1] / df['EntryPrice'].iloc[0] -1
    expected_fwd_custom_ret_idx_0 = (df['CustomPrice'].iloc[1] / df['EntryPrice'].iloc[0]) - 1
    assert np.isclose(res_custom_price_fwd['forward_return_01'].iloc[0], expected_fwd_custom_ret_idx_0) 