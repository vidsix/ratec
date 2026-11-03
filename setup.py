from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "ratec._rcengine",
            sources=["ratec/_rcengine.c"],
            extra_compile_args=[
                "-O3", "-march=native",
                # mandatory for bit-exactness with the reference pipeline
                "-ffp-contract=off", "-fno-fast-math",
            ],
            libraries=["m", "pthread"],
        )
    ]
)
