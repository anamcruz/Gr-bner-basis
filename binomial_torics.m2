-*
Binomial and Toric Ideals Generator

This script generates binomial or toric ideals and its correspondent Gröbner
Basis plus computation info, regarding polynomial additions. They are generated
using th software system Macaulay2, http://www2.macaulay2.com/Macaulay2/.

This was achieved by using the "Ideals.m2" and "SelectionStrategies.m2" packages
created by Dylan Peifer. They can be found in the code repository on Github's
Dylan Peifer account, https://github.com/dylanpeifer/deepgroebner.git. This
script was created by modifying the "make_stats.m2" script in order to save
additional information and calculate the Gröbner Basis using the Buchberger's
Algorithm from "SelectionStrategies.m2".

The command to generate a set of binomial and toric ideals have the format, for
binomials and torics:
M2 --script binomial_torics.m2 <number of variables>-<maximum
degree>-<number of ideal generators>-<distribution> <set size>

M2 --script binomial_torics.m2 toric-<number of variables>-<lower bound
degree of monomials>-<maximum bound degree of monomials>-<number of monomials>
<set size>

For example:
    M2 --script binomial_torics.m2 3-20-10-uniform 1000
    M2 --script binomial_torics.m2 toric-2-0-5-8 1000

*-


needsPackage("Ideals", FileName => "m2/Ideals.m2")
needsPackage("SelectionStrategies", FileName => "m2/SelectionStrategies.m2")

capitalize = method()
capitalize String := s -> toUpper s#0 | substring(1, #s, s)


    parseIdealDist = method()
    parseIdealDist String := HashTable => dist -> (
        -- Return HashTable with parameters for ideal distribution.
        args := separate("-", dist);
        params := {};
        if  args#0 == "toric" then (
        params = {"kind" => "toric",
                  "n" => value(args#1),
                  "L" => value(args#2),
                  "U" => value(args#3),
                  "M" => value(args#4)};
        )
        else (
        params = {"kind" => "binom",
                  "n" => value(args#0),
                  "d" => value(args#1),
                  "s" => value(args#2),
                  "degs" => args#3,
                  "consts" => member("consts", args),
                  "homog" => member("homog", args),
                  "pure" => member("pure", args)};
            );
        hashTable params
        )

    setupOutFile1 = method()
    setupOutFile1 String := String => dist -> (
        -- Setup output file and return its name.
        directory := "data/stats/test-" | dist | "/Ideals" | "/";
        if not isDirectory directory then makeDirectory directory;
        outFileIdeals := directory | dist | currentTime() | ".csv";

        F = openOutAppend outFileIdeals;
        close F;
        outFileIdeals
        )

    setupOutFile2 = method()
    setupOutFile2 String := String => dist -> (
        -- Setup output file and return its name.
        directory := "data/stats/test-" | dist | "/GB" | "/";
        if not isDirectory directory then makeDirectory directory;
        outFileGB := directory | dist | currentTime() | ".csv";

        E = openOutAppend outFileGB;
        close E;
        outFileGB
        )

    setupOutFile3 = method()
    setupOutFile3 String := String => dist -> (
        -- Setup output file and return its name.
        directory := "data/stats/test-" | dist | "/Stats" | "/";
        if not isDirectory directory then makeDirectory directory;
        outFileStats := directory | dist | currentTime() | ".csv";
        -*zeroReductions, nonzeroReductions, polynomialAdditions,
        monomialAdditions, degree, dim, regularity*-
        P = openOutAppend outFileStats;
        close P;
        outFileStats
        )


    setupOutFile4 = method()
    setupOutFile4 String := String => dist -> (
        -- Setup output file and return its name.
        directory := "data/stats/test-" | dist | "/Degree" | "/";
        if not isDirectory directory then makeDirectory directory;
        outFileDG := directory | dist | currentTime() | ".csv";

        P = openOutAppend outFileDG;
        close P;
        outFileDG
        )




    dist = scriptCommandLine#1;
    samples = value(scriptCommandLine#2);
    if #scriptCommandLine == 4 then setRandomSeed(value(scriptCommandLine#3));

    outFileIdeals = setupOutFile1 dist;
    outFileGB = setupOutFile2 dist;
    outFileDG = setupOutFile4 dist;
    outFileStats = setupOutFile3 dist;
    H = parseIdealDist dist;


    if H#"kind" == "binom" then (

        R = QQ[vars(0..(H#"n" - 1))];
        opts = new OptionTable from {Constants => H#"consts",
                                     Degrees => capitalize H#"degs",
                                     Homogeneous => H#"homog",
                                     Pure => false};

        F := openOutAppend outFileIdeals;
        E := openOutAppend outFileGB;
        P := openOutAppend outFileDG;
        L := openOutAppend outFileStats;

        for sample from 1 to samples do (

            I = randomBinomialIdeal(R, H#"d", H#"s", opts);

            (G, stats) := buchberger(I);
            s := toString first entries gens I;

            if isHomogeneous I then a = regularity I else a = regularity ideal leadTerm I;

            g := toString G;


            zero := toString stats#"zeroReductions";

            nonzero := toString stats#"nonzeroReductions";

            polyadd := toString stats#"polynomialAdditions";

            monoadd := toString stats#"monomialAdditions";

            sdegree := toString degree I;

            sdim := toString dim I;

            sr := toString a;

            F << s << endl;

            E << g << endl;

            L << zero << "," << nonzero << "," << polyadd << "," << monoadd <<
            "," << sdegree << "," << sdim << "," << sr << endl;


            for ngerador from 0 to H#"s"-1 do(
                idealasmatrix = first entries gens I;
                EXPONENTS = exponents idealasmatrix#ngerador;

                P << EXPONENTS << ",";
            );
            P << endl;
            print sample;
            );

        close F;
        close E;
        close L;
        close P;
    ) else (
        R = QQ[vars(0..(H#"M" - 1))];
        F := openOutAppend outFileIdeals;
        E := openOutAppend outFileGB;
        L := openOutAppend outFileStats;
        P := openOutAppend outFileDG;
        for sample from 1 to samples do (

            I = randomToricIdeal(H#"n", H#"L", H#"U", H#"M");
            DEGREE = degrees I;
            (G, stats) := buchberger(I);
            s := toString first entries gens I;

            if isHomogeneous I then a = regularity I else a = regularity ideal leadTerm I;

            g := toString G;


            zero := toString stats#"zeroReductions";

            nonzero := toString stats#"nonzeroReductions";

            polyadd := toString stats#"polynomialAdditions";

            monoadd := toString stats#"monomialAdditions";

            sdegree := toString degree I;

            sdim := toString dim I;

            sr := toString a;


            F << s << endl;

            E << g << endl;

            L << zero << "," << nonzero << "," << polyadd << "," << monoadd <<
            "," << sdegree << "," << sdim << "," << sr << endl;

            idealasmatrix = gens I;
            aux = numgens source idealasmatrix;
            for ngerador from 0 to aux-1 do(
                idealhash = first entries gens I;
                EXPONENTS = exponents idealhash#ngerador;

                P << EXPONENTS << ",";
            );
            P << endl;

            print sample;
            );

        close F;
        close E;
        close L;
        close P;
    );