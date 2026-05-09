import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import io
    import polars as pl
    import py7zr

    return io, mo, pl, py7zr, pyprojroot


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    qcew_path = root_path / 'data' / 'qcew'
    return binary_path, qcew_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QCEW binaries
    """)
    return


@app.cell
def _(pl):
    qcew_dtype_dict = {
        'area_fips': pl.String,
        'own_code': pl.String,
        'industry_code': pl.String,
        'agglvl_code': pl.String,
        'size_code': pl.String,
        'year': pl.Int16,
        'qtr': pl.String,
        'disclosure_code': pl.String,
        'annual_avg_estabs': pl.Float32,
        'annual_avg_emplvl': pl.Float32,
        'total_annual_wages': pl.Float32
    }
    qcew_cols_list = list(qcew_dtype_dict.keys())
    return qcew_cols_list, qcew_dtype_dict


@app.cell
def _(io):
    # Define a custom buffer that prevents py7zr from closing it prematurely
    class MemoryBufferIO(io.BytesIO):
        def size(self):
            """
            py7zr expects a .size() method on the IO object.
            """
            return self.getbuffer().nbytes
    
        def close(self):
            """
            Block py7zr from closing the buffer so Polars can still read it.
            """
            pass 
        
        def force_close(self):
            """
            A method to actually clear the memory when we are done.
            """
            super().close()

    return (MemoryBufferIO,)


@app.cell
def _(MemoryBufferIO, py7zr):
    # Define the Factory class that py7zr will call to create the buffers
    class PolarsMemoryFactory(py7zr.io.WriterFactory):
        def __init__(self):
            self.buffers = {}

        def create(self, filename: str):
            bio = MemoryBufferIO()
            self.buffers[filename] = bio
            return bio

    return (PolarsMemoryFactory,)


@app.cell
def _(
    PolarsMemoryFactory,
    binary_path,
    pl,
    py7zr,
    qcew_cols_list,
    qcew_dtype_dict,
    qcew_path,
):
    for t in range(2005, 2018):
        print(t)
        sevenz_path = qcew_path / f"{t}_annual_singlefile.7z"
        target_csv = f'{t}.annual.singlefile.csv'
        mem_factory = PolarsMemoryFactory()

        # Extract straight into memory
        with py7zr.SevenZipFile(sevenz_path, mode='r') as zf:
            zf.extract(targets=[target_csv], factory=mem_factory)
    
        extracted_buffer = mem_factory.buffers[target_csv]

        # Rewind the buffer to the beginning
        extracted_buffer.seek(0)

        # Read straight from the RAM buffer using Polars
        qcew_df = pl.read_csv(
            extracted_buffer,
            columns=qcew_cols_list,
            schema_overrides=qcew_dtype_dict
        )
    
        # Write to Parquet using Polars
        qcew_df.write_parquet(binary_path / f'qcew_{t}.parquet')
    
        # Flush the memory buffer before the next iteration
        extracted_buffer.force_close()
    return


if __name__ == "__main__":
    app.run()
