import numpy as np

single_spin_ops = {
    '1/2': {
        'x': 0.5 * np.array(
            [[0, 1],
             [1, 0]]
        ),
        
        'y': -0.5 * 1j * np.array(
            [[0, 1],
            [-1, 0]]
        ),
        
        'z': 0.5 * np.array(
            [[1, 0],
            [0, -1]]
        ),
        
        'p': np.array(
            [[0, 1],
            [0, 0]]
        ),
        
        'm': np.array(
            [[0, 0],
            [1, 0]],
        )
    },
    
    '1': {
        'x': (1 / np.sqrt(2)) * np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]
        ),
        
        'y': -(1 / np.sqrt(2)) * 1j * np.array([
            [0, 1, 0],
            [-1, 0, 1],
            [0, -1, 0]
        ]),
        
        'z': np.array(
            [[1, 0, 0],
            [0, 0, 0],
            [0, 0, -1]]
        ),
        
        'p': np.sqrt(2) * np.array(
            [[0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]]
        ),
        
        'm': np.sqrt(2) * np.array(
            [[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]]
        )
    },
    
    '3/2': {
        'x': 0.5 * np.array([
            [0, np.sqrt(3), 0, 0],
            [np.sqrt(3), 0, 2, 0],
            [0, 2, 0, np.sqrt(3)],
            [0, 0, np.sqrt(3), 0]
        ]),
        
        'y': -0.5 * 1j * np.array([
            [0, np.sqrt(3), 0, 0],
            [-np.sqrt(3), 0, 2, 0],
            [0, -2, 0, np.sqrt(3)],
            [0, 0, -np.sqrt(3), 0]
        ]),
        
        'z': np.array([
            [3/2, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, -1/2, 0],
            [0, 0, 0, -3/2]
        ]),
        
        'p': np.array([
            [0, np.sqrt(3), 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, np.sqrt(3)],
            [0, 0, 0, 0]
        ]),
        
        'm': np.array([
            [0, 0, 0, 0],
            [np.sqrt(3), 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, np.sqrt(3), 0]
        ])
    },
    
    '2': {
        'x': 0.5 * np.array([
            [0, 2, 0, 0, 0],
            [2, 0, np.sqrt(6), 0, 0],
            [0, np.sqrt(6), 0, np.sqrt(6), 0],
            [0, 0, np.sqrt(6), 0, 2],
            [0, 0, 0, 2, 0]
        ]),
        
        'y': -0.5 * 1j * np.array([
            [0, 2, 0, 0, 0],
            [-2, 0, np.sqrt(6), 0, 0],
            [0, -np.sqrt(6), 0, np.sqrt(6), 0],
            [0, 0, -np.sqrt(6), 0, 2],
            [0, 0, 0, -2, 0]
        ]),
        
        'z': np.array([
            [2, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0],
            [0, 0, 0, 0, -2]
        ]),
        
        'p': np.array([
            [0, 2, 0, 0, 0],
            [0, 0, np.sqrt(6), 0, 0],
            [0, 0, 0, np.sqrt(6), 0],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0]
        ]),
        
        'm': np.array([
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, np.sqrt(6), 0, 0, 0],
            [0, 0, np.sqrt(6), 0, 0],
            [0, 0, 0, 2, 0]
        ])
    }
}