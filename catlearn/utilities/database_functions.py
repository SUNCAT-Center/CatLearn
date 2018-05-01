"""Functions to create databases storing feature matrix."""
import sqlite3
from sqlite3 import IntegrityError
import numpy as np


class DescriptorDatabase(object):
    """Store sets of descriptors for a given atoms object assigned a unique ID.

    The descriptors for a given system can be stored in the ase.atoms object,
    though we typically find this method to be slower.
    """

    def __init__(self, db_name='descriptor_store.sqlite', table='Descriptors'):
        """Setup for the database class.

        Parameters
        ----------
        db_name : str
            Name of the sqlite database to connect. Default is
            descriptor_store.sqlite.
        table : str
            Name of the table in which to store the descriptors. Default is
            Descriptors.
        """
        self.db_name = db_name
        self.table = table
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def create_db(self, names):
        """Function to setup a database storing descriptors.

        Parameters
        ----------
        names : list
            List of heading names for features and targets.
        """
        cstring = 'uuid text'
        for i in names:
            cstring += ', %s float' % i

        # Create a table
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS %(table)s
                            (%(cstring)s)""" % {'table': self.table,
                                                'cstring': cstring})

    def create_column(self, new_column):
        """Function to create a new column in the table.

        The new column will be initialized with None values.

        Parameters
        ----------
        new_column : str
            Name of new feature or target.
        """
        for c in new_column:
            self.cursor.execute("alter table %(table)s add column \
                                %(new_column)s 'float'" % {'table': self.table,
                                                           'new_column': c})
            self.conn.commit()

    def fill_db(self, descriptor_names, data):
        """Function to fill the descriptor database.

        Parameters
        ----------
        descriptor_names : list
            List of descriptor names for features and targets.
        data : array
            First row should contain string of UUIDs, thereafter array should
            contain floats corresponding to the descriptor names provided.
        """
        qm = '(?'
        for i in range(len(descriptor_names)):
            qm += ',?'
        qm += ')'

        self.cursor.executemany("INSERT INTO %(table)s VALUES %(qm)s"
                                % {'table': self.table, 'qm': qm}, data)
        self.conn.commit()

    def update_descriptor(self, descriptor, new_data, unique_id):
        """Function to update a descriptor based on a given uuid.

        Parameters
        ----------
        descriptor : str
            Name of descriptor to be updated.
        new_data : float
            New value to be entered into table.
        unique_id : str
            The UUID of the entry to be updated.
        """
        sql = """
        UPDATE %(table)s
        SET %(descriptor)s = %(new_data)s
        WHERE uuid = '%(unique_id)s'
        """ % {'table': self.table, 'descriptor': descriptor,
               'new_data': str(new_data), 'unique_id': unique_id}
        self.cursor.execute(sql)
        self.conn.commit()

    def query_db(self, unique_id=None, names=None):
        """Return single row based on uuid or all rows.

        Parameters
        ----------
        unique_id : str
            If specified, the data corresponding to the given UUID will be
            returned. If None, all rows will be returned.
        names : list
            If specified, only the data corresponding to provided column names
            will be returned. If None, all columns will be returned.
        """
        if names is None:
            names = '*'
        else:
            d = names[0]
            if len(names) > 1:
                for i in names[1:]:
                    d += ', %s' % i
            names = d

        data = []
        if unique_id is None:
            for row in self.cursor.execute("SELECT %(desc)s FROM %(table)s"
                                           % {'desc': names,
                                              'table': self.table}):
                data.append(row)
        else:
            sql = "SELECT %(desc)s FROM %(table)s WHERE uuid=?" \
                % {'desc': names, 'table': self.table}
            self.cursor.execute(sql, [('%s' % unique_id)])
            data = self.cursor.fetchall()[0]

        return np.asarray(data)

    def get_column_names(self):
        """Function to get the of a supplied table column names."""
        cursor = self.conn.execute('select * from %s' % self.table)

        return [description[0] for description in cursor.description]


class FingerprintDB():
    """A class for accessing a temporary SQLite database.

    This function works as a context manager and should be used as follows:

    with FingerprintDB() as fpdb:
        (Perform operation here)

    This syntax will automatically construct the temporary database, or access
    an existing one. Upon exiting the indentation, the changes to the database
    will be automatically commited.
    """

    def __init__(self, db_name='fingerprints.db', verbose=False):
        """Initialize the class.

        Parameters
        ----------
        db_name : str
            Name of the database file to access. Will connect to
            'fingerprints.db' by default.
        verbose : bool
            Will print additional information.
        """
        self.db_name = db_name
        self.verbose = verbose

    def __enter__(self):
        """Called whenever the class is used with a 'with' statement."""
        self.con = sqlite3.connect(self.db_name)
        self.c = self.con.cursor()
        self.create_table()

        return self

    def __exit__(self, type, value, tb):
        """Upon exiting the 'with' statement, __exit__ is called."""
        self.con.commit()
        self.con.close()

    def create_table(self):
        """Create the database table framework used in SQLite.

        This includes 3 tables: images, parameters, and fingerprints.

        The images table currently stores ase_id information and
        a unqiue string. This can be adapted in the future to support
        atoms objects.

        The parameters table stores a symbol (10 character maximum)
        for convenient reference and a description of the parameter.

        The fingerprints table holds a unique image and parmeter ID
        along with a float value for each. The ID pair must be unique.
        """
        self.c.execute("""CREATE TABLE IF NOT EXISTS images(
        iid INTEGER PRIMARY KEY AUTOINCREMENT,
        ase_id CHAR(32) UNIQUE NOT NULL,
        identity TEXT
        )""")

        self.c.execute("""CREATE TABLE IF NOT EXISTS parameters(
        pid INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol CHAR(10) UNIQUE NOT NULL,
        description TEXT
        )""")

        self.c.execute("""CREATE TABLE IF NOT EXISTS fingerprints(
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INT NOT NULL,
        param_id INT NOT NULL,
        value REAL,
        FOREIGN KEY(image_id) REFERENCES images(image_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
        FOREIGN KEY(param_id) REFERENCES parameters(param_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
        UNIQUE(image_id, param_id)
        )""")

    def image_entry(self, asedb_entry=None, identity=None):
        """Enter a single ase-db image into the fingerprint database.

        This table can be expanded to contain atoms objects in the future.

        Parameters
        ----------
        d : object
            An ase-db object which can be parsed.
        identity : str
            An identifier of the users choice.

        Returns
        -------
        d.id : int
            The ase ID colleted for the ase-db object.
        """
        atoms = asedb_entry.toatoms()

        # ase-db ID with identity must be unique. If not, it will be skipped.
        try:
            self.c.execute("""INSERT INTO images (ase_id, identity)
            VALUES(?, ?)""", (asedb_entry.unique_id, identity))
        except(IntegrityError):
            if self.verbose:
                print('ASE ID with identifier already defined: {} {}'.format(
                    asedb_entry.id,
                    identity))

        return d.id

    def parameter_entry(self, symbol=None, description=None):
        """Function for entering unique parameters into the database.

        Paramters
        ---------
        symbol : str
            A unique symbol the entry can be referenced by. If None, the symbol
            will be the ID of the parameter as a string.
        description : str
            A description of the parameter.
        """
        # If no symbol is provided, use the parameter ID (str).
        if not symbol:
            self.c.execute("""SELECT MAX(pid) FROM parameters""")
            symbol = str(int(self.c.fetchone()[0]) + 1)

        # The symbol must be unique. If not, it will be skipped.
        try:
            self.c.execute("""INSERT INTO parameters (symbol, description)
            VALUES(?, ?)""", (symbol, description))
        except(IntegrityError):
            if self.verbose:
                print('Symbol already defined: {}'.format(symbol))

        # Each instance needs to be commited to ensure no overwriting.
        # This could potentially result in slowdown.
        self.con.commit()

    def get_parameters(self, selection=None, display=False):
        """Return integer values corresponding to parameter IDs.

        The array returned will be for a set of provided symbols. If no
        selection is provided, return all symbols.

        Parameters
        ----------
        selection : list
            List of symbols in parameters table to be selected.
        display : bool
            If True, print parameter descriptions.

        Returns
        -------
        res : array
            Return the integer values of selected parameters.
        """
        # If no selection is made, return all parameters.
        if not selection:
            self.c.execute("""SELECT pid, symbol, description
            FROM parameters""")
            res = self.c.fetchall()
        else:
            res = []
            for i, s in enumerate(selection):
                self.c.execute("""SELECT pid, symbol, description
                FROM parameters WHERE symbol = ?""", (s,))
                res += [self.c.fetchone()]

        if display:
            print('[ID ]: key    - Description')
            print('---------------------------')
            for r in res:
                print('[{0:^3}]: {1:<10} - {2}'.format(*r))

        return np.array(res).T[0].astype(int)

    def fingerprint_entry(self, ase_id, param_id, value):
        """Enter fingerprint value to database for given ase and parameter ID.

        Parameters
        ----------
        ase_id : int
            The ase unique ID associated with an atoms object in the database.
        param_id : int or str
            The parameter ID or symbol associated with and entry in the
            paramters table.
        value : float
            The value of the parameter for the atoms object.
        """
        # If parameter symbol is given, get the ID
        if isinstance(param_id, str):
            self.c.execute("""SELECT pid FROM parameters
            WHERE symbol = ?""", (param_id,))
            param_id = self.c.fetchone()

            if param_id:
                param_id = param_id[0]
            else:
                raise(KeyError, 'parameter symbol not found')

        self.c.execute("""SELECT iid FROM images
        WHERE ase_id = ?""", (ase_id,))
        image_id = self.c.fetchone()[0]

        try:
            self.c.execute("""INSERT INTO fingerprints (image_id, param_id, value)
            VALUES(?, ?, ?)""", (str(image_id), int(param_id), float(value)))
        except(IntegrityError):
            if self.verbose:
                print('Fingerprint already defined: {}, {}, {}'.format(
                    image_id, param_id, value))

    def get_fingerprints(self, ase_ids, params=[]):
        """Return values of provided parameters for each ase_id provided.

        Parameters
        ----------
        ase_id : list
            The ase ID(s) associated with an atoms object in the database.
        params : list
            List of symbols or int in parameters table to be selected.

        Returns
        -------
        fingerprint : array
            An array of values associated with the given parameters (a
            fingerprint) for each ase_id.
        """
        if isinstance(params, np.ndarray):
            params = params.tolist()

        if not params or isinstance(params[0], str):
            params = self.get_parameters(selection=params)
            psel = ','.join(params.astype(str))
        elif isinstance(params[0], int):
            psel = ','.join(np.array(params).astype(str))

        if isinstance(ase_ids, np.ndarray):
            ase_ids = ase_ids.tolist()

        asel = tuple(np.array(ase_ids).astype(str))

        self.c.execute("""SELECT GROUP_CONCAT(value) FROM fingerprints
        JOIN images on fingerprints.image_id = images.iid
        WHERE param_id IN ({}) AND ase_id IN {}
        GROUP BY ase_id""".format(psel, asel))
        fetch = self.c.fetchall()

        fingerprint = np.zeros((len(ase_ids), len(params)))
        for i, f in enumerate(fetch):
            fingerprint[i] = f[0].split(',')

        return fingerprint
