import streamlit as st


def density_speed_documentation() -> None:
    st.write(
        r"""
                ## Density:
            The measurement method involves several steps to calculate the individual density based on proximity data. The process is as follows:

            - Calculate Half Distances: Mathematically, this can be expressed as:

            $$
            H = \frac{1}{2}  \left( \text{dist}_{\text{prev}} +  \text{dist}_{\text{next}}\right),
            $$

            where $H$ represents the half sum of distances, $\text{dist}_{\text{prev}}$ and $\text{dist}_{\text{next}}$ denote the Euklidean distances to the previous and next entities, respectively.

            - Calculate Individual Density: The individual density ($\rho$) is then calculated as the inverse of the half sum of distances:
            $$
            \rho = \frac{1}{H}
            $$

            ## Speed
            The calculation of speed is based on the displacement in the $x$ and $y$ directions over time. The method involves the following steps:
            - Calculate Displacements: The displacement in both the $x$ and $y$ directions is calculated as the difference between successive positions, accounting for displacement over a specified number of frames ($\Delta t$). This is done separately for each entity, identified by its $id$. The mathematical expressions for these displacements are:
            """
    )
    st.latex(
        r"""
        \begin{align*}
        \Delta x &= x(t + \Delta t) - x(t). \\
        \Delta y &= y(t + \Delta t) - y(t).
        \end{align*}
        """
    )
    st.write(
        """
        where $\\Delta x$ and $\\Delta y$ represent the displacements in the $x$ and $y$ directions, respectively, and $\\Delta t$ is the difference in frame indices used for the calculation.

        - Compute Distance Traveled: The distance traveled between the frames is computed using the Pythagorean theorem, which combines the displacements in both directions:
        $$
        \\text{distance} = \\sqrt{\\Delta x^2 + \\Delta y^2}.
        $$
        - Calculate Speed: Finally, the speed is calculated as the ratio of the distance traveled to the time
        """
    )
    st.latex(
        r"""
        \begin{equation}
        \text{speed} = \frac{\text{distance}}{\Delta t}
        \end{equation}
        """
    )
    st.write(
        """
        This yields the speed of each entity between the specified frames, taking into account the displacements in both spatial dimensions.
        See [implementation here](https://github.com/PedestrianDynamics/gender-experiments/blob/main/analysis.py#L10).
        """
    )
