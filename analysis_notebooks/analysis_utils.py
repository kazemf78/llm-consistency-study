
# import re
# def prep(input_df: pd.DataFrame, ltx=False, input_pct=False) -> pd.DataFrame:
#     def shorten_model_name(name):
#         return name.split("/")[-1] if "/" in name else name
#     df = input_df.copy()
#     try:
#         df['model'] = df['model'].apply(shorten_model_name)
#     except:
#         df = df.reset_index().rename(columns={"index": "model"})
#         df['model'] = df['model'].apply(shorten_model_name)
#     # df = df.rename(
#     #     columns={col: col.replace("_", "-") for col in df.columns}
#     # )
#     if ltx:
#         df.columns = [
#             re.sub(r"_", " ", col).strip()#.title()
#             for col in df.columns
#         ]
#     num_cols = df.select_dtypes(include='number').columns
#     if not input_pct:
#         mask = df[num_cols].abs() <= 1
#         df[num_cols] = df[num_cols].where(~mask, df[num_cols] * 100)
    
#     return df

import re
import pandas as pd # Ensure pandas is imported
def prep(input_df: pd.DataFrame, ltx=False, input_pct=False, ignore_pct_col_pats=None) -> pd.DataFrame:
    def shorten_model_name(name):
        return name.split("/")[-1] if "/" in name else name
    
    # 1. Setup and Renaming
    df = input_df.copy()
    try:
        df['model'] = df['model'].apply(shorten_model_name)
    except:
        df = df.reset_index().rename(columns={"index": "model"})
        df['model'] = df['model'].apply(shorten_model_name)
    
    if ltx:
        df.columns = [
            re.sub(r"_", " ", col).strip()
            for col in df.columns
        ]
        
    # 2. Identify Numeric Columns
    num_cols = df.select_dtypes(include='number').columns
    
    # Exclude ignored columns
    if ignore_pct_col_pats is not None:
        assert isinstance(ignore_pct_col_pats, list), "ignore_pct_col_pats should be a list of regex patterns."
        num_cols = [
            col for col in num_cols
            if not any(re.search(pat.lower(), col.lower()) for pat in ignore_pct_col_pats)
        ]

    # 3. Handle Percentage Conversion (Column-level)
    if not input_pct:
        # Determine which numeric columns should be multiplied based on column-wide condition.
        cols_to_scale = []

        for col in num_cols:
            col_abs = df[col].abs()

            # Condition: ALL values in column have abs <= 1
            if (col_abs <= 1).all():
                cols_to_scale.append(col)

        # Multiply entire columns
        df[cols_to_scale] = df[cols_to_scale] * 100


    # --- YOUR NEW LOGIC STARTS HERE ---

    def conditional_round(x):
        """
        Applies conditional rounding based on the magnitude of x.
        If x >= 1 (or x <= -1), round to 1 decimal place.
        If 0 < x < 1 (or -1 < x < 0), round to 2 decimal places.
        """
        if pd.isna(x):
            return x
        
        # Check absolute value for the condition
        if abs(x) >= 1:
            # Above 1 (e.g., 92.34 -> 92.3)
            return round(x, 1)
        else:
            # Below 1 (e.g., 0.234 -> 0.23)
            return round(x, 2)
            
    # Apply the custom rounding function to all numeric columns
    num_cols = df.select_dtypes(include='number').columns
    # df[num_cols] = df[num_cols].applymap(conditional_round)
    # --- COLUMN-BASED ROUNDING LOGIC ---
    # display(df)

    for col in num_cols:
        col_abs = df[col].abs()

        if (col_abs >= 1).any():
            # Column contains values >= 1 → round whole column to 1 decimal
            df[col] = df[col].round(1)
        else:
            # Column values all < 1 → round whole column to 2 decimals
            df[col] = df[col].round(2)

    # num_cols = df.select_dtypes(include='number').columns
    # df[num_cols] = df[num_cols].astype(str)
    # df[num_cols] = df[num_cols].applymap(lambda x: f"{x}")
    # display(df)
    # --- END NEW LOGIC ---
    
    return df


import pandas as pd

def prep_for_paper(df, rename_map, order_cols=None):#, round_decimals=1):
    """
    Clean, rename, reorder, and round columns for paper-ready tables.
    """
    df2 = df.copy()

    # Rename columns for paper
    df2 = df2.rename(columns=rename_map)

    # Reorder columns if specified
    if order_cols is not None:
        df2 = df2[order_cols]

    return df2

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def df_to_acl_icml_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    short_model_names: dict = None,
    float_format: str = "%.1f",
    col_format: str = None,
    full_width: bool = False,
    resize: bool = True,
    output_path: str = None,
    multiline_headers: bool = True,
    max_header_len: int = 20,
    custom_float_format: bool = False,
    bold_cells=None,      # NEW FEATURE
    bold_negative_cols=None,
    color_negative_cols=None,
    column_rename_map=None,
) -> str:

    def latex_multiline(text, align="c", max_len=20):
        words = text.split()
        lines, current = [], []
        for w in words:
            if len(" ".join(current + [w])) > max_len:
                lines.append(" ".join(current))
                current = [w]
            else:
                current.append(w)
        if current:
            lines.append(" ".join(current))
        return rf"\makecell[{align}]{{" + r"\\ ".join(lines) + "}"

    df2 = df.copy()

    # Apply model name shortening
    if short_model_names is not None and 'model' in df2.columns:
        df2['model'] = df2['model'].replace(short_model_names)

    # Parse alignment
    if col_format is not None:
        col_aligns = [c for c in col_format if c in "lcr"]
    else:
        col_aligns = ["c"] * df2.shape[1]

    # Multi-line column headers
    if multiline_headers:
        new_cols = []
        for i, c in enumerate(df2.columns):
            align = col_aligns[i] if i < len(col_aligns) else "c"
            if len(c) > max_header_len:
                new_cols.append(latex_multiline(c, align=align, max_len=max_header_len))
            else:
                new_cols.append(c)
        df2.columns = new_cols

    # === NEW: BOLD FORMATTING PRE-PROCESSING ===
    bold_mask = pd.DataFrame(False, index=df2.index, columns=df2.columns)

    # REMOVE for proper modularity! (we can use the exact same process for each style instead of such operations!)
    color_mask = pd.DataFrame(False, index=df2.index, columns=df2.columns)
    if bold_negative_cols is not None:
        for col in bold_negative_cols:
            if col in df2.columns:
                col_vals = pd.to_numeric(df2[col], errors="coerce")
                bold_mask[col] |= col_vals < 0
    if color_negative_cols:
        for col in color_negative_cols:
            if col in df2.columns:
                vals = pd.to_numeric(df2[col], errors="coerce")
                color_mask[col] |= vals < 0
    ###


    if bold_cells is not None:

        # Case A — dict, e.g. {"colname": "max"} or {"colname": ["max","min"]}
        if isinstance(bold_cells, dict):
            for col, rule in bold_cells.items():
                if col not in df2.columns:
                    continue

                col_vals = pd.to_numeric(df2[col], errors="coerce")

                # allow rule to be a single string or a list/tuple of rules
                rules = rule if isinstance(rule, (list, tuple)) else [rule]

                col_mask = pd.Series(False, index=df2.index)
                for r in rules:
                    if r == "max":
                        col_mask |= (col_vals == col_vals.max())
                    elif r == "min":
                        col_mask |= (col_vals == col_vals.min())
                    else:
                        raise ValueError(f"Unknown rule '{r}' in bold_cells for column {col}")

                bold_mask[col] |= col_mask


        # Case B — callable → Series or DataFrame mask
        elif callable(bold_cells):
            mask = bold_cells(df2)

            # Case B1 — DataFrame mask → align per column & index
            if isinstance(mask, pd.DataFrame):
                # Only consider columns that exist in df2
                for col in mask.columns:
                    if col in bold_mask.columns:
                        col_mask = (
                            mask[col]
                            .reindex(df2.index)
                            .fillna(False)
                            .astype(bool)
                        )
                        bold_mask[col] |= col_mask

            # Case B2 — Series mask → apply only to that column (mask.name)
            elif isinstance(mask, pd.Series):
                # print('here')
                # print(mask.name)
                if mask.name not in df2.columns:
                    raise ValueError(
                        f"Callable returned a Series named '{mask.name}', "
                        f"but this is not a valid column in the DataFrame."
                    )
                # bold_mask[mask.name] |= mask
                col_mask = (
                    mask.reindex(df2.index).fillna(False).astype(bool)
                )
                bold_mask[mask.name] |= col_mask                
                

            else:
                raise ValueError("Callable bold_cells must return a Series or DataFrame.")

        # Case C — set of (row, col)
        elif isinstance(bold_cells, set):
            for r, c in bold_cells:
                if c in df2.columns and r in df2.index:
                    bold_mask.at[r, c] = True

    num_cols = df2.select_dtypes(include='number').columns
    df2 = df2.astype(object)
    # === APPLY BOLDING TO CELL CONTENTS ===
    for col in df2.columns:
        for i in df2.index:
            # if bold_mask.at[i, col]:
            #     df2.at[i, col] = f"\\textbf{{{df2.at[i, col]}}}"

            # REMOVE for proper modularity!
            cell = df2.at[i, col]
            if bold_mask.at[i, col]:
                cell = f"\\textbf{{{cell}}}"
            if color_mask.at[i, col]:
                cell = f"\\textcolor{{red}}{{{cell}}}"
            df2.at[i, col] = cell
            ###

    # Optional float formatting override
    if custom_float_format:
        df2[num_cols] = df2[num_cols].astype(str)
    if column_rename_map:
        df2 = df2.rename(columns=column_rename_map)

    latex_body = df2.to_latex(
        index=False,
        float_format=float_format,
        column_format=col_format,
        escape=False
    )

    env = "table*" if full_width else "table"
    width_key = r"\textwidth" if full_width else r"\linewidth"

    header = f"\\begin{{{env}}}[t]\n\\centering\n"
    footer = f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\end{{{env}}}\n"

    if resize:
        latex = header + f"\\resizebox{{{width_key}}}{{!}}{{%\n" + latex_body + "}\n" + footer
    else:
        latex = header + latex_body + footer

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)
    return latex



model_name_map = {
    # OpenAI
    "gpt-4o": "GPT-4o",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1-Mini",
    "gpt-3.5-turbo": "GPT-3.5",

    "openai/gpt-oss-20b": "GPT-OSS 20B",

    "gpt-oss-20b": "GPT-OSS 20B",

    # LLaMA
    "meta-llama/Meta-Llama-3-8B-Instruct": "LLaMA-3 8B",
    "meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1 8B",

    "Meta-Llama-3-8B-Instruct": "LLaMA-3 8B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1 8B",

    # Qwen 2.5
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5 7B",
    "Qwen2.5-7B-Instruct": "Qwen-2.5 7B",

    # Qwen 3 (base)
    "Qwen/Qwen3-0.6B": "Qwen-3 0.6B",
    "Qwen/Qwen3-1.7B": "Qwen-3 1.7B",
    "Qwen/Qwen3-4B": "Qwen-3 4B",
    "Qwen/Qwen3-8B": "Qwen-3 8B",
    "Qwen/Qwen3-14B": "Qwen-3 14B",
    "Qwen/Qwen3-32B": "Qwen-3 32B",

    "Qwen3-0.6B": "Qwen-3 0.6B",
    "Qwen3-1.7B": "Qwen-3 1.7B",
    "Qwen3-4B": "Qwen-3 4B",
    "Qwen3-8B": "Qwen-3 8B",
    "Qwen3-14B": "Qwen-3 14B",
    "Qwen3-32B": "Qwen-3 32B",

    # Qwen 3 (+ thinking)
    "Qwen/Qwen3-0.6B[with_thinking]": "Qwen-3 0.6B +Thinking",
    "Qwen/Qwen3-1.7B[with_thinking]": "Qwen-3 1.7B +Thinking",
    "Qwen/Qwen3-4B[with_thinking]": "Qwen-3 4B +Thinking",
    "Qwen/Qwen3-8B[with_thinking]": "Qwen-3 8B +Thinking",
    "Qwen/Qwen3-14B[with_thinking]": "Qwen-3 14B +Thinking",
    "Qwen/Qwen3-32B[with_thinking]": "Qwen-3 32B +Thinking",

    "Qwen3-0.6B[with_thinking]": "Qwen-3 0.6B +Thinking",
    "Qwen3-1.7B[with_thinking]": "Qwen-3 1.7B +Thinking",
    "Qwen3-4B[with_thinking]": "Qwen-3 4B +Thinking",
    "Qwen3-8B[with_thinking]": "Qwen-3 8B +Thinking",
    "Qwen3-14B[with_thinking]": "Qwen-3 14B +Thinking",
    "Qwen3-32B[with_thinking]": "Qwen-3 32B +Thinking",

}